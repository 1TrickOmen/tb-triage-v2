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
    predict_metadata_probabilities,
    summarize_prediction_metrics,
)


def parse_thresholds(value: str) -> list[float]:
    if not value.strip():
        return list(DEFAULT_THRESHOLDS)
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved MobileNetV2 model on an external metadata CSV.')
    parser.add_argument('--repo-root', default='.', help='Path to repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', required=True, help='External evaluation metadata CSV')
    parser.add_argument('--run-dir', required=True, help='Training run directory containing mobilenetv2_baseline.keras')
    parser.add_argument('--output-dir', default='', help='Defaults to <run-dir>/external_eval/<metadata_stem>')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.40, help='Operating threshold applied to prob_tb')
    parser.add_argument('--thresholds', default='', help='Optional sweep. Defaults to the standard threshold grid.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    run_dir = (repo_root / args.run_dir).resolve()
    model_path = run_dir / 'mobilenetv2_baseline.keras'
    if not model_path.exists():
        raise FileNotFoundError(f'Saved model not found at {model_path}')
    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found at {metadata_csv}')

    output_dir = (repo_root / args.output_dir).resolve() if args.output_dir else (run_dir / 'external_eval' / metadata_csv.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = predict_metadata_probabilities(
        str(model_path),
        str(metadata_csv),
        image_size=(args.image_size, args.image_size),
    )
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

    summary = summarize_prediction_metrics(predictions_df, threshold=args.threshold)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    threshold_df = evaluate_thresholds(predictions_df, parse_thresholds(args.thresholds))
    threshold_df.to_csv(output_dir / 'threshold_metrics.csv', index=False)

    confusion_df = pd.DataFrame([
        {'actual': 'Normal', 'predicted': 'Normal', 'count': summary['tn']},
        {'actual': 'Normal', 'predicted': 'TB', 'count': summary['fp']},
        {'actual': 'TB', 'predicted': 'Normal', 'count': summary['fn']},
        {'actual': 'TB', 'predicted': 'TB', 'count': summary['tp']},
    ])
    confusion_df.to_csv(output_dir / 'confusion_at_threshold.csv', index=False)

    print('external_eval_metrics')
    print(json.dumps(summary, indent=2))
    print('threshold_metrics')
    print(threshold_df.to_string(index=False))
    print('saved_to', output_dir)


if __name__ == '__main__':
    main()
