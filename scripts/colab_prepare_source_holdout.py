from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.data_utils import SEED  # noqa: E402
from src.data.source_holdout import make_source_holdout_metadata, write_summary_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a source-held-out metadata CSV for Colab training/evaluation.')
    parser.add_argument('--repo-root', default='.', help='Path to the repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', default='data/processed/merged_metadata.csv')
    parser.add_argument('--holdout-source', required=True, help='Source to reserve as true unseen test source, e.g. montgomery, shenzhen, tbx11k')
    parser.add_argument('--output-metadata-csv', default='', help='Defaults to data/processed/source_holdout/<holdout>_experiment_metadata.csv')
    parser.add_argument('--output-holdout-csv', default='', help='Defaults to data/processed/source_holdout/<holdout>_holdout_only.csv')
    parser.add_argument('--output-summary-json', default='', help='Defaults to data/processed/source_holdout/<holdout>_summary.json')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    if not metadata_csv.exists():
        legacy_metadata_csv = (repo_root / 'data/merged_metadata.csv').resolve()
        if args.metadata_csv == 'data/processed/merged_metadata.csv' and legacy_metadata_csv.exists():
            metadata_csv = legacy_metadata_csv
        else:
            raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')

    base_dir = repo_root / 'data' / 'processed' / 'source_holdout'
    output_metadata_csv = (repo_root / args.output_metadata_csv).resolve() if args.output_metadata_csv else (base_dir / f'{args.holdout_source}_experiment_metadata.csv')
    output_holdout_csv = (repo_root / args.output_holdout_csv).resolve() if args.output_holdout_csv else (base_dir / f'{args.holdout_source}_holdout_only.csv')
    output_summary_json = (repo_root / args.output_summary_json).resolve() if args.output_summary_json else (base_dir / f'{args.holdout_source}_summary.json')

    summary = make_source_holdout_metadata(
        str(metadata_csv),
        holdout_source=args.holdout_source,
        output_metadata_csv=str(output_metadata_csv),
        output_holdout_csv=str(output_holdout_csv),
        seed=args.seed,
    )
    write_summary_json(summary, str(output_summary_json))

    print('source_holdout_summary')
    print(json.dumps(summary, indent=2, default=str))
    print('saved_metadata', output_metadata_csv)
    print('saved_holdout_only', output_holdout_csv)
    print('saved_summary', output_summary_json)


if __name__ == '__main__':
    main()
