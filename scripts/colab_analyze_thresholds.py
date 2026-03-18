from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.thresholds import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    evaluate_thresholds,
    predict_test_probabilities,
    select_threshold_for_target_recall,
)


def parse_thresholds(value: str) -> list[float]:
    if not value.strip():
        return list(DEFAULT_THRESHOLDS)
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze TB decision thresholds from saved model probabilities.')
    parser.add_argument('--repo-root', default='.', help='Path to repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', default='data/processed/merged_metadata.csv')
    parser.add_argument('--run-dir', required=True, help='Directory containing mobilenetv2_baseline.keras and optional test_predictions.csv')
    parser.add_argument('--predictions-csv', default='', help='Optional explicit predictions CSV with y_true and prob_tb columns')
    parser.add_argument('--output-dir', default='', help='Defaults to <run-dir>/threshold_analysis')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--thresholds', default='', help='Comma-separated TB thresholds, e.g. 0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--target-recall', type=float, default=None, help='Optional target TB recall. Script will print the best threshold meeting it.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = (repo_root / args.run_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve() if args.output_dir else (run_dir / 'threshold_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)

    predictions_path = ((repo_root / args.predictions_csv).resolve() if args.predictions_csv else (run_dir / 'test_predictions.csv'))
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
    else:
        model_path = run_dir / 'mobilenetv2_baseline.keras'
        if not model_path.exists():
            raise FileNotFoundError(f'Could not find predictions CSV or saved model at {model_path}')
        metadata_csv = (repo_root / args.metadata_csv).resolve()
        predictions_df = predict_test_probabilities(
            str(model_path),
            str(metadata_csv),
            image_size=(args.image_size, args.image_size),
        )
        predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    threshold_df = evaluate_thresholds(predictions_df, thresholds)
    threshold_df.to_csv(output_dir / 'threshold_metrics.csv', index=False)

    best_for_target = None
    if args.target_recall is not None:
        best_for_target = select_threshold_for_target_recall(threshold_df, args.target_recall)
        with open(output_dir / 'target_recall_selection.json', 'w') as f:
            json.dump({
                'target_recall': args.target_recall,
                'selected_threshold': best_for_target,
            }, f, indent=2)

    print('threshold_metrics')
    print(threshold_df.to_string(index=False))
    print('saved_to', output_dir)
    if best_for_target is None and args.target_recall is not None:
        print(f'no_threshold_met_target_recall {args.target_recall}')
    elif best_for_target is not None:
        print('target_recall_selection', json.dumps(best_for_target, indent=2))


if __name__ == '__main__':
    main()
