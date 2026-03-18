from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.data_utils import resolve_metadata_paths  # noqa: E402


VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def _flatten_tensors(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true_f, y_pred_f = _flatten_tensors(y_true, y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    return dice_coefficient(y_true, y_pred, smooth=smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred, smooth=smooth)


def jaccard_index(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true_f, y_pred_f = _flatten_tensors(y_true, y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    return jaccard_index(y_true, y_pred, smooth=smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    return jaccard_index(y_true, y_pred, smooth=smooth)


def jaccard_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    return 1.0 - jaccard_index(y_true, y_pred, smooth=smooth)


def get_segmentation_custom_objects() -> dict[str, object]:
    return {
        'dice_coefficient': dice_coefficient,
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss,
        'jaccard_index': jaccard_index,
        'iou': iou,
        'iou_score': iou_score,
        'jaccard_loss': jaccard_loss,
        'K': K,
    }


def load_model(model_path: Path):
    return tf.keras.models.load_model(model_path, custom_objects=get_segmentation_custom_objects(), compile=False)


def get_model_input_channels(model: tf.keras.Model) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    channels = input_shape[-1]
    if channels not in (1, 3):
        raise ValueError(f'Unsupported segmentation model input shape: {input_shape}. Expected ...x1 or ...x3 input.')
    return int(channels)


def preprocess_image(image_bgr: np.ndarray, image_size: tuple[int, int], channels: int) -> np.ndarray:
    resized = cv2.resize(image_bgr, image_size)
    if channels == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return (gray.astype('float32') / 255.0)[..., np.newaxis]

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype('float32') / 255.0


def postprocess_mask(mask_pred: np.ndarray, output_size: tuple[int, int], threshold: float) -> np.ndarray:
    mask = np.squeeze(mask_pred)
    mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_LINEAR)
    mask = (mask >= threshold).astype(np.uint8) * 255
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict lung masks from a saved segmentation model and write mask paths back to metadata.')
    parser.add_argument('--repo-root', default='.', help='Path to the repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', default='data/processed/merged_metadata.csv')
    parser.add_argument('--segmentation-model', required=True, help='Path to a saved Keras segmentation model (.keras)')
    parser.add_argument('--output-masks-dir', default='data/processed/predicted_lung_masks')
    parser.add_argument('--output-metadata-csv', default='data/processed/merged_metadata_with_predicted_masks.csv')
    parser.add_argument('--image-size', type=int, default=512, help='Segmentation model input size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold applied to model outputs')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    output_masks_dir = (repo_root / args.output_masks_dir).resolve()
    output_metadata_csv = (repo_root / args.output_metadata_csv).resolve()
    model_path = (repo_root / args.segmentation_model).resolve()

    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')
    if not model_path.exists():
        raise FileNotFoundError(f'Segmentation model not found at {model_path}')

    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    input_channels = get_model_input_channels(model)
    df = pd.read_csv(metadata_csv)
    df = resolve_metadata_paths(df, str(metadata_csv))

    mask_paths: list[str] = []
    generated = 0
    skipped = 0

    for _, row in df.iterrows():
        image_path = Path(str(row['image_path']))
        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS or not image_path.exists():
            mask_paths.append(str(row.get('mask_path', '') or ''))
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            mask_paths.append(str(row.get('mask_path', '') or ''))
            skipped += 1
            continue

        original_h, original_w = image_bgr.shape[:2]
        image_input = preprocess_image(image_bgr, (args.image_size, args.image_size), input_channels)
        prediction = model.predict(np.expand_dims(image_input, axis=0), verbose=0)[0]
        mask = postprocess_mask(prediction, (original_w, original_h), args.threshold)

        relative_mask_path = Path(args.output_masks_dir) / f"{Path(str(row['image_id'])).stem}.png"
        absolute_mask_path = repo_root / relative_mask_path
        absolute_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(absolute_mask_path), mask)

        mask_paths.append(str(relative_mask_path))
        generated += 1

    df['mask_path'] = mask_paths
    df.to_csv(output_metadata_csv, index=False)

    print('mask_generation_summary', json.dumps({
        'rows': int(len(df)),
        'generated_masks': int(generated),
        'skipped_rows': int(skipped),
        'output_masks_dir': str(output_masks_dir),
        'output_metadata_csv': str(output_metadata_csv),
    }, indent=2))


if __name__ == '__main__':
    main()
