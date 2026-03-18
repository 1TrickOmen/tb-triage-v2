from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.data_utils import resolve_metadata_paths  # noqa: E402


VALID_MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _resolve_optional_path(value: str, repo_root: Path, metadata_dir: Path) -> Path | None:
    if not value or str(value).strip() in {'', 'nan', 'None'}:
        return None
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return p
    for candidate in (metadata_dir / p, repo_root / p, Path.cwd() / p):
        if candidate.exists():
            return candidate.resolve()
    return None


def _find_mask_for_row(row: pd.Series, repo_root: Path, metadata_dir: Path, masks_dir: Path | None) -> Path | None:
    direct = _resolve_optional_path(str(row.get('mask_path', '') or ''), repo_root, metadata_dir)
    if direct is not None:
        return direct
    if masks_dir is None:
        return None
    stem = Path(str(row['image_id'])).stem
    for ext in VALID_MASK_EXTENSIONS:
        candidate = masks_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Create masked input images and a derived metadata CSV for the segmentation value test.')
    parser.add_argument('--repo-root', default='.', help='Path to repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', default='data/processed/merged_metadata.csv')
    parser.add_argument('--output-metadata-csv', default='data/processed/merged_metadata_masked.csv')
    parser.add_argument('--output-images-dir', default='data/processed/masked_images')
    parser.add_argument('--masks-dir', default='', help='Optional directory of masks named by image_id stem, e.g. foo/IMG_001.png')
    parser.add_argument('--mask-threshold', type=int, default=127, help='Binary threshold for grayscale masks in [0,255]')
    parser.add_argument('--allow-missing-masks', action='store_true', help='If set, keep original image_path for rows without a usable mask instead of failing.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    output_metadata_csv = (repo_root / args.output_metadata_csv).resolve()
    output_images_dir = (repo_root / args.output_images_dir).resolve()
    masks_dir = (repo_root / args.masks_dir).resolve() if args.masks_dir else None
    metadata_dir = metadata_csv.parent.resolve()

    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')
    if masks_dir is not None and not masks_dir.exists():
        raise FileNotFoundError(f'Masks directory not found at {masks_dir}')

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    df = resolve_metadata_paths(df, str(metadata_csv))

    new_image_paths: list[str] = []
    resolved_mask_paths: list[str] = []
    notes: list[str] = []
    masked_count = 0
    missing_mask_count = 0

    for _, row in df.iterrows():
        image_path = Path(str(row['image_path']))
        mask_path = _find_mask_for_row(row, repo_root, metadata_dir, masks_dir)
        row_notes = str(row.get('notes', '') or '')

        if not image_path.exists():
            raise FileNotFoundError(f'Image missing for {row.get("image_id", "unknown")}: {image_path}')

        if mask_path is None:
            if not args.allow_missing_masks:
                raise FileNotFoundError(
                    f"No mask found for image_id={row.get('image_id', 'unknown')} image_path={image_path}. "
                    "Provide complete masks or rerun with --allow-missing-masks if you intentionally want fallback behavior."
                )
            new_image_paths.append(str(row['image_path']))
            resolved_mask_paths.append(str(row.get('mask_path', '') or ''))
            notes.append((row_notes + ';mask_missing_for_masked_variant').strip(';'))
            missing_mask_count += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise ValueError(f'Could not read image at {image_path}')
        if mask_gray is None:
            if not args.allow_missing_masks:
                raise ValueError(
                    f'Could not read mask for image_id={row.get("image_id", "unknown")} at {mask_path}. '
                    'Fix the mask file or rerun with --allow-missing-masks to keep the raw image for that row.'
                )
            new_image_paths.append(str(row['image_path']))
            resolved_mask_paths.append(str(mask_path))
            notes.append((row_notes + ';mask_unreadable_for_masked_variant').strip(';'))
            missing_mask_count += 1
            continue

        mask_resized = cv2.resize(mask_gray, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized >= args.mask_threshold).astype(np.uint8)
        masked_bgr = image_bgr * mask_binary[:, :, None]

        relative_output_path = Path(args.output_images_dir) / f"{Path(str(row['image_id'])).stem}.png"
        absolute_output_path = repo_root / relative_output_path
        absolute_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(absolute_output_path), masked_bgr)

        new_image_paths.append(str(relative_output_path))
        resolved_mask_paths.append(str(mask_path.relative_to(repo_root)) if mask_path.is_relative_to(repo_root) else str(mask_path))
        notes.append((row_notes + ';masked_input_generated').strip(';'))
        masked_count += 1

    df['image_path'] = new_image_paths
    df['mask_path'] = resolved_mask_paths
    df['notes'] = notes
    df.to_csv(output_metadata_csv, index=False)

    print('masked_metadata_summary', json.dumps({
        'rows': int(len(df)),
        'masked_images_written': int(masked_count),
        'rows_missing_masks': int(missing_mask_count),
        'output_images_dir': str(output_images_dir),
        'output_metadata_csv': str(output_metadata_csv),
    }, indent=2))


if __name__ == '__main__':
    main()
