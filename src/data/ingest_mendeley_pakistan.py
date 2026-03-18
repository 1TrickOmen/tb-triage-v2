from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

FIELDNAMES = [
    'image_id', 'patient_id', 'source_dataset', 'label_raw', 'label_final',
    'image_path', 'mask_path', 'view_position', 'width', 'height', 'split',
    'is_external_test', 'include_for_training', 'bbox_count', 'notes'
]

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
TB_DIR_HINTS = {'tb', 'tuberculosis', 'positive', 'tb_patient', 'patients'}
NORMAL_DIR_HINTS = {'normal', 'healthy', 'negative', 'control', 'controls'}


def _iter_images(root: Path):
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _label_from_path(path: Path, tb_root: Path | None, normal_root: Path | None) -> tuple[str, str]:
    parts = {part.lower() for part in path.parts}

    if tb_root is not None and tb_root in path.parents:
        return 'tb', 'TB'
    if normal_root is not None and normal_root in path.parents:
        return 'normal', 'Normal'

    if parts & TB_DIR_HINTS:
        return 'tb', 'TB'
    if parts & NORMAL_DIR_HINTS:
        return 'normal', 'Normal'
    return '', ''


def build_records(dataset_root: str, *, tb_dir: str = '', normal_dir: str = ''):
    root = Path(dataset_root).resolve()
    tb_root = (root / tb_dir).resolve() if tb_dir else None
    normal_root = (root / normal_dir).resolve() if normal_dir else None

    if not root.exists():
        raise FileNotFoundError(f'Dataset root not found: {root}')
    if tb_dir and not tb_root.exists():
        raise FileNotFoundError(f'TB directory not found: {tb_root}')
    if normal_dir and not normal_root.exists():
        raise FileNotFoundError(f'Normal directory not found: {normal_root}')

    records: list[dict[str, str]] = []
    label_counter = Counter()
    skipped = 0

    for image_path in _iter_images(root):
        label_raw, label_final = _label_from_path(image_path, tb_root, normal_root)
        if not label_final:
            skipped += 1
            continue

        relative_image_path = image_path.relative_to(root.parent) if root.parent in image_path.parents else image_path
        image_id = image_path.name
        patient_id = image_path.stem
        records.append({
            'image_id': image_id,
            'patient_id': patient_id,
            'source_dataset': 'mendeley_pakistan_tb_cxr',
            'label_raw': label_raw,
            'label_final': label_final,
            'image_path': str(relative_image_path),
            'mask_path': '',
            'view_position': '',
            'width': '',
            'height': '',
            'split': 'external_eval',
            'is_external_test': 'true',
            'include_for_training': 'false',
            'bbox_count': '',
            'notes': 'external_validation_only',
        })
        label_counter[label_final] += 1

    return records, label_counter, skipped


def save_csv(records: list[dict[str, str]], output_csv: str) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build metadata CSV for the Pakistan Mendeley TB chest X-ray dataset.')
    parser.add_argument('--dataset-root', required=True, help='Root directory containing the extracted dataset files')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--tb-dir', default='', help='Optional dataset-root-relative folder containing TB-positive images')
    parser.add_argument('--normal-dir', default='', help='Optional dataset-root-relative folder containing normal images')
    args = parser.parse_args()

    records, label_counter, skipped = build_records(
        args.dataset_root,
        tb_dir=args.tb_dir,
        normal_dir=args.normal_dir,
    )
    if not records:
        raise ValueError(
            'No labeled images found. Pass --tb-dir and --normal-dir explicitly if the folder names are not obvious.'
        )
    save_csv(records, args.output_csv)
    print('records', len(records))
    print('label_counts', dict(label_counter))
    print('skipped_unlabeled_files', skipped)
    print('output_csv', args.output_csv)
