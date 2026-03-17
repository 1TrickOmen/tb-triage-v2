from __future__ import annotations

import csv
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable

TB_TAGS = {"active_tb", "latent_tb", "active_tb&latent_tb"}
NORMAL_TAGS = {"healthy"}
EXCLUDE_TAGS = {"sick_but_non-tb", "uncertain_tb"}
FIELDNAMES = [
    'image_id', 'patient_id', 'source_dataset', 'label_raw', 'label_final',
    'image_path', 'mask_path', 'view_position', 'width', 'height', 'split',
    'is_external_test', 'include_for_training', 'bbox_count', 'notes'
]


def _map_tag_to_label(tag_name: str):
    if tag_name in TB_TAGS:
        return 'TB', True
    if tag_name in NORMAL_TAGS:
        return 'Normal', True
    if tag_name in EXCLUDE_TAGS:
        return '', False
    return '', False


def _iter_annotation_members(tar: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tar:
        if member.isfile() and member.name.endswith('.json') and '/ann/' in member.name:
            yield member


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


def parse_records(tar_path: str, extracted_paths: dict[str, str] | None = None):
    label_raw_counter = Counter()
    label_final_counter = Counter()
    records = []
    with tarfile.open(tar_path) as tar:
        for member in _iter_annotation_members(tar):
            f = tar.extractfile(member)
            if f is None:
                continue
            ann = json.load(f)
            tags = ann.get('tags', [])
            if not tags:
                continue
            tag_name = tags[0].get('name', '')
            label_final, include = _map_tag_to_label(tag_name)
            split = member.name.split('/')[0]
            file_name = Path(member.name).name.replace('.json', '')
            image_member_path = f'{split}/img/{file_name}'
            size = ann.get('size', {})
            bbox_count = len(ann.get('objects', []))
            image_path = extracted_paths.get(image_member_path, image_member_path) if extracted_paths else image_member_path
            record = {
                'image_id': file_name,
                'patient_id': file_name.rsplit('.', 1)[0],
                'source_dataset': 'tbx11k',
                'label_raw': tag_name,
                'label_final': label_final,
                'image_path': image_path,
                'mask_path': '',
                'view_position': '',
                'width': size.get('width', ''),
                'height': size.get('height', ''),
                'split': split,
                'is_external_test': 'true' if split == 'test' else 'false',
                'include_for_training': 'true' if include and split != 'test' else 'false',
                'bbox_count': bbox_count,
                'notes': '' if include else 'excluded_from_v2_binary_baseline',
            }
            records.append(record)
            label_raw_counter[tag_name] += 1
            label_final_counter[label_final or 'EXCLUDED'] += 1
    return records, label_raw_counter, label_final_counter


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
    records, raw_counts, final_counts = parse_records(args.tar_path, extracted_paths=extracted_paths)
    save_csv(records, args.output_csv)
    print('records', len(records))
    print('raw_counts', dict(raw_counts))
    print('final_counts', dict(final_counts))
    print('output_csv', args.output_csv)
