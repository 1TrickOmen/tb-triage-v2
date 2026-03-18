from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.source_balance import add_source_balanced_sample_weights, write_summary_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Add source-balanced training sample weights to an explicit-split metadata CSV.')
    parser.add_argument('--repo-root', default='.', help='Path to the repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', required=True, help='Metadata CSV that already contains explicit train/val/test splits')
    parser.add_argument('--output-metadata-csv', default='', help='Defaults to <metadata_stem>_source_balanced.csv next to the input CSV')
    parser.add_argument('--output-summary-json', default='', help='Defaults to <metadata_stem>_source_balanced_summary.json next to the output CSV')
    parser.add_argument('--split-column', default='experiment_split')
    parser.add_argument('--weight-column', default='sample_weight')
    parser.add_argument('--balance-mode', choices=['source_label', 'normal_by_source'], default='source_label', help='source_label equalizes each (source,label) group; normal_by_source only rebalances Normal rows across sources.')
    parser.add_argument('--normal-weight-multiplier', type=float, default=1.0, help='Extra multiplier applied to Normal-row weights. Keep at 1.0 for the cleanest first run.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')

    output_metadata_csv = (repo_root / args.output_metadata_csv).resolve() if args.output_metadata_csv else metadata_csv.with_name(f'{metadata_csv.stem}_source_balanced.csv')
    output_summary_json = (repo_root / args.output_summary_json).resolve() if args.output_summary_json else output_metadata_csv.with_name(f'{output_metadata_csv.stem}_summary.json')

    summary = add_source_balanced_sample_weights(
        str(metadata_csv),
        str(output_metadata_csv),
        split_column=args.split_column,
        weight_column=args.weight_column,
        balance_mode=args.balance_mode,
        normal_weight_multiplier=args.normal_weight_multiplier,
    )
    write_summary_json(summary, str(output_summary_json))

    print('source_balanced_summary')
    print(json.dumps(summary, indent=2, default=str))
    print('saved_metadata', output_metadata_csv)
    print('saved_summary', output_summary_json)


if __name__ == '__main__':
    main()
