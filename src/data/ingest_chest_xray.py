from __future__ import annotations

import csv
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable

FIELDNAMES = [
    'image_id', 'patient_id', 'source_dataset', 'label_raw', 'label_final',
    'image_path', 'mask_path', 'view_position', 'width', 'height', 'split',
    'is_external_test', 'include_for_training', 'bbox_count', 'notes'
]


def extract_images(tar_path: str, output_root: str) -> dict[str, str]:
    extracted_paths: dict[str, str] = {}
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile() or '/img/' not in member.name:
                continue

            target = output_dir / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open('wb') as dst:
                    dst.write(source.read())
            extracted_paths[member.name] = str(target)

    return extracted_paths


def _iter_image_members(tar: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tar:
        if member.isfile() and '/img/' in member.name:
            yield member


def _dataset_from_image_id(image_id: str) -> str:
    if image_id.startswith('MCUCXR_'):
        return 'montgomery'
    if image_id.startswith('CHNCXR_'):
        return 'shenzhen'
    return 'chest_xray'


def _label_from_image_id(image_id: str) -> tuple[str, str]:
    stem = image_id.rsplit('.', 1)[0]
    suffix = stem.rsplit('_', 1)[-1]
    if suffix == '1':
        return '1', 'TB'
    if suffix == '0':
        return '0', 'Normal'
    return '', ''


def parse_records(tar_path: str, extracted_paths: dict[str, str] | None = None):
    source_counter = Counter()
    label_counter = Counter()
    records = []

    with tarfile.open(tar_path) as tar:
        for member in _iter_image_members(tar):
            split = member.name.split('/')[0]
            image_id = Path(member.name).name
            image_member_path = member.name
            label_raw, label_final = _label_from_image_id(image_id)
            source_dataset = _dataset_from_image_id(image_id)
            image_path = extracted_paths.get(image_member_path, image_member_path) if extracted_paths else image_member_path

            record = {
                'image_id': image_id,
                'patient_id': image_id.rsplit('.', 1)[0],
                'source_dataset': source_dataset,
                'label_raw': label_raw,
                'label_final': label_final,
                'image_path': image_path,
                'mask_path': '',
                'view_position': '',
                'width': '',
                'height': '',
                'split': 'train' if split == 'CXRpng-train' else split,
                'is_external_test': 'false',
                'include_for_training': 'true',
                'bbox_count': '',
                'notes': 'lung_masks_available_only_inside_archive_annotations',
            }
            records.append(record)
            source_counter[source_dataset] += 1
            label_counter[label_final or 'EMPTY'] += 1

    return records, source_counter, label_counter


def save_csv(records, output_csv: str):
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tar-path', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--extract-root', default='')
    args = parser.parse_args()

    extracted_paths = extract_images(args.tar_path, args.extract_root) if args.extract_root else None
    records, source_counts, label_counts = parse_records(args.tar_path, extracted_paths=extracted_paths)
    save_csv(records, args.output_csv)
    print('records', len(records))
    print('source_counts', dict(source_counts))
    print('label_counts', dict(label_counts))
    print('output_csv', args.output_csv)
