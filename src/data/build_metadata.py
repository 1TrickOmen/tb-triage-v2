from __future__ import annotations

import csv
from pathlib import Path

from .ingest_chest_xray import extract_images as extract_chest_images
from .ingest_chest_xray import parse_records as parse_chest_records
from .ingest_tbx11k import extract_images as extract_tbx11k_images
from .ingest_tbx11k import parse_records as parse_tbx11k_records

FIELDNAMES = [
    'image_id', 'patient_id', 'source_dataset', 'label_raw', 'label_final',
    'image_path', 'mask_path', 'view_position', 'width', 'height', 'split',
    'is_external_test', 'include_for_training', 'bbox_count', 'notes'
]


def _write_csv(records: list[dict[str, str]], output_csv: str) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def build_metadata(
    tbx11k_tar: str,
    chest_xray_tar: str,
    extract_root: str,
    merged_output_csv: str,
    tbx11k_output_csv: str | None = None,
    chest_output_csv: str | None = None,
) -> dict[str, int]:
    extract_root_path = Path(extract_root)
    tbx11k_paths = extract_tbx11k_images(tbx11k_tar, str(extract_root_path / 'tbx11k'))
    chest_paths = extract_chest_images(chest_xray_tar, str(extract_root_path / 'chest_xray'))

    tbx11k_records, _, _ = parse_tbx11k_records(tbx11k_tar, extracted_paths=tbx11k_paths)
    chest_records, _, _ = parse_chest_records(chest_xray_tar, extracted_paths=chest_paths)
    merged_records = chest_records + tbx11k_records

    if chest_output_csv:
        _write_csv(chest_records, chest_output_csv)
    if tbx11k_output_csv:
        _write_csv(tbx11k_records, tbx11k_output_csv)
    _write_csv(merged_records, merged_output_csv)

    return {
        'merged_records': len(merged_records),
        'chest_records': len(chest_records),
        'tbx11k_records': len(tbx11k_records),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tbx11k-tar', required=True)
    parser.add_argument('--chest-xray-tar', required=True)
    parser.add_argument('--extract-root', required=True)
    parser.add_argument('--merged-output-csv', required=True)
    parser.add_argument('--tbx11k-output-csv', default='')
    parser.add_argument('--chest-output-csv', default='')
    args = parser.parse_args()

    summary = build_metadata(
        tbx11k_tar=args.tbx11k_tar,
        chest_xray_tar=args.chest_xray_tar,
        extract_root=args.extract_root,
        merged_output_csv=args.merged_output_csv,
        tbx11k_output_csv=args.tbx11k_output_csv or None,
        chest_output_csv=args.chest_output_csv or None,
    )
    for key, value in summary.items():
        print(key, value)
