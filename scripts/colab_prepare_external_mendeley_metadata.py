from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_mendeley_pakistan import build_records, save_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare metadata CSV for the Pakistan Mendeley TB external evaluation dataset.')
    parser.add_argument('--repo-root', default='.', help='Path to repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--dataset-root', required=True, help='Dataset root containing the extracted images')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--tb-dir', default='', help='Optional dataset-root-relative TB folder')
    parser.add_argument('--normal-dir', default='', help='Optional dataset-root-relative normal folder')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_csv = (repo_root / args.output_csv).resolve()

    records, label_counter, skipped = build_records(
        str(dataset_root),
        tb_dir=args.tb_dir,
        normal_dir=args.normal_dir,
    )
    if not records:
        raise ValueError('No labeled images found. Folder naming is probably different; pass --tb-dir and --normal-dir explicitly.')

    save_csv(records, str(output_csv))
    summary = {
        'dataset_root': str(dataset_root),
        'records': len(records),
        'label_counts': dict(label_counter),
        'skipped_unlabeled_files': int(skipped),
        'output_csv': str(output_csv),
    }
    print('external_metadata_summary')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
