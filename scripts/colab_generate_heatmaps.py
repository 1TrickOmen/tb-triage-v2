from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.data_utils import resolve_metadata_paths  # noqa: E402
from src.explainability.gradcam import get_gradcam_heatmap, overlay_heatmap_on_image  # noqa: E402


LABEL_NAMES = ['Normal', 'TB']
ARCH_TO_FUNCTION = {
    'mobilenetv2': 'build_mobilenetv2',
    'densenet121': 'build_densenet121',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for a sample of metadata rows or specific image paths.')
    parser.add_argument('--repo-root', default='.', help='Path to the repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--model-path', required=True, help='Path to the saved Keras model (.keras) relative to repo root')
    parser.add_argument('--metadata-csv', required=True, help='Metadata CSV containing image_path and label_final columns')
    parser.add_argument('--output-dir', required=True, help='Directory where side-by-side heatmap previews will be written')
    parser.add_argument('--architecture', choices=sorted(ARCH_TO_FUNCTION), required=True, help='Model architecture used for training')
    parser.add_argument('--image-size', type=int, default=256, help='Resize images to this square size before inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--samples-per-class', type=int, default=5, help='How many Normal and TB images to sample when --image-path is not provided')
    parser.add_argument('--image-path', dest='image_paths', action='append', default=[], help='Optional metadata image_path value (repeatable). If set, sampling is skipped.')
    return parser.parse_args()


def extract_last_conv_layer_name(architecture: str) -> str:
    models_py = REPO_ROOT / 'src' / 'classification' / 'models.py'
    module = ast.parse(models_py.read_text())
    function_name = ARCH_TO_FUNCTION[architecture]

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for inner in node.body:
                if isinstance(inner, ast.Assign):
                    for target in inner.targets:
                        if isinstance(target, ast.Name) and target.id == 'last_conv_layer_name':
                            value = inner.value
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                return value.value
            break

    raise ValueError(f'Could not extract last_conv_layer_name for architecture={architecture!r} from {models_py}')


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if 'label_final' not in df.columns or 'image_path' not in df.columns:
        raise ValueError('Metadata CSV must contain label_final and image_path columns.')
    df = df[df['label_final'].isin(LABEL_NAMES)].copy()
    if df.empty:
        raise ValueError('Metadata CSV does not contain any Normal/TB rows.')
    df = resolve_metadata_paths(df, str(metadata_csv))
    df['image_exists'] = df['image_path'].apply(lambda p: Path(p).exists())
    df = df[df['image_exists']].copy()
    if df.empty:
        raise ValueError('No existing image files were found after resolving metadata paths.')
    return df.reset_index(drop=True)


def sanitize_name(value: str) -> str:
    keep = [c if c.isalnum() or c in {'-', '_', '.'} else '_' for c in value]
    return ''.join(keep).strip('._') or 'image'


def select_rows(df: pd.DataFrame, *, image_paths: list[str], samples_per_class: int, seed: int) -> pd.DataFrame:
    if image_paths:
        requested_keys: dict[str, str] = {}
        for raw in image_paths:
            raw = str(raw)
            requested_keys[str(Path(raw))] = raw
            requested_keys[str(Path(raw).resolve())] = raw

        selected_parts = []
        matched_inputs: set[str] = set()
        for _, row in df.iterrows():
            raw_path = str(row['image_path'])
            normalized_path = str(Path(raw_path).resolve())
            matched_raw = requested_keys.get(raw_path) or requested_keys.get(normalized_path)
            if matched_raw is not None:
                selected_parts.append(row.to_dict())
                matched_inputs.add(matched_raw)

        missing = [str(path) for path in image_paths if str(path) not in matched_inputs]
        if missing:
            raise FileNotFoundError(
                'Some --image-path values were not found in the metadata after path resolution: ' + ', '.join(missing)
            )
        return pd.DataFrame(selected_parts).reset_index(drop=True)

    rng = random.Random(seed)
    sampled_parts = []
    for label in LABEL_NAMES:
        label_df = df[df['label_final'] == label].copy()
        if label_df.empty:
            raise ValueError(f'No {label} rows found in metadata.')
        take = min(samples_per_class, len(label_df))
        indices = list(label_df.index)
        rng.shuffle(indices)
        sampled_parts.append(label_df.loc[indices[:take]])
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_image_for_model(image_path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f'Failed to read image: {image_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size)
    model_input = resized.astype('float32') / 255.0
    return resized, np.expand_dims(model_input, axis=0)


def build_side_by_side(original_rgb: np.ndarray, overlay_rgb: np.ndarray) -> np.ndarray:
    divider = np.full((original_rgb.shape[0], 12, 3), 255, dtype=np.uint8)
    return np.concatenate([original_rgb, divider, overlay_rgb], axis=1)


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    model_path = (repo_root / args.model_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    last_conv_layer_name = extract_last_conv_layer_name(args.architecture)
    model = tf.keras.models.load_model(model_path)
    metadata_df = load_metadata(metadata_csv)
    selected_df = select_rows(
        metadata_df,
        image_paths=args.image_paths,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    manifest_rows: list[dict[str, object]] = []
    image_size = (args.image_size, args.image_size)

    for row_idx, row in selected_df.iterrows():
        image_path = Path(row['image_path']).resolve()
        original_rgb, model_input = load_image_for_model(image_path, image_size)

        predictions = model.predict(model_input, verbose=0)[0]
        pred_index = int(np.argmax(predictions))
        pred_label = LABEL_NAMES[pred_index]
        heatmap = get_gradcam_heatmap(model_input, model, last_conv_layer_name, pred_index=pred_index)
        overlay_bgr = overlay_heatmap_on_image(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        preview_rgb = build_side_by_side(original_rgb, overlay_rgb)

        image_id = str(row.get('image_id', image_path.name))
        output_name = f"{row_idx + 1:02d}_{sanitize_name(str(row['label_final']))}_{sanitize_name(image_id)}.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

        manifest_rows.append({
            'output_path': str(output_path),
            'image_id': image_id,
            'patient_id': row.get('patient_id', ''),
            'source_dataset': row.get('source_dataset', ''),
            'label_final': row['label_final'],
            'image_path': str(image_path),
            'predicted_label': pred_label,
            'prob_normal': float(predictions[0]),
            'prob_tb': float(predictions[1]),
            'last_conv_layer_name': last_conv_layer_name,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / 'manifest.csv', index=False)

    summary = {
        'model_path': str(model_path),
        'metadata_csv': str(metadata_csv),
        'output_dir': str(output_dir),
        'architecture': args.architecture,
        'last_conv_layer_name': last_conv_layer_name,
        'images_written': int(len(manifest_rows)),
        'label_counts': manifest_df['label_final'].value_counts().to_dict() if len(manifest_df) else {},
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('heatmap_summary')
    print(json.dumps(summary, indent=2))
    if len(manifest_df):
        print('manifest')
        print(manifest_df[['image_id', 'label_final', 'predicted_label', 'prob_tb', 'output_path']].to_string(index=False))


if __name__ == '__main__':
    main()
