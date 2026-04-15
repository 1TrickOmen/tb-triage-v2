from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def main() -> None:
    parser = argparse.ArgumentParser(description='Colab-friendly TB baseline runner.')
    parser.add_argument('--repo-root', default='.', help='Path to the repo root inside Colab, e.g. /content/tb-triage-v2')
    parser.add_argument('--metadata-csv', default='data/processed/merged_metadata.csv')
    parser.add_argument('--output-dir', default='experiments/colab-baseline')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--trainable-base', action='store_true')
    parser.add_argument('--trainable-fraction', type=float, default=None, help='Fraction of backbone layers to unfreeze from the end, e.g. 0.25')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none', help='Apply training-set class weighting. Use balanced to upweight TB if classes are skewed.')
    parser.add_argument('--architecture', choices=['mobilenetv2', 'densenet121'], default='mobilenetv2', help='Which CNN architecture to train.')
    parser.add_argument('--augmentation', choices=['mild', 'strong'], default='mild', help='Use mild unless you are intentionally stress-testing augmentation.')
    parser.add_argument('--rebuild-metadata', action='store_true', help='Rebuild merged metadata from uploaded dataset tar files before training.')
    parser.add_argument('--tbx11k-tar', default='data/raw/tbx11k/tbx11k-DatasetNinja.tar')
    parser.add_argument('--chest-xray-tar', default='data/raw/chest-xray/chest-xray-masks-and-labels-DatasetNinja.tar')
    parser.add_argument('--extract-root', default='data/processed/extracted')
    args = parser.parse_args()

    from src.data.build_metadata import build_metadata
    from src.classification.train import train_baseline_from_metadata

    repo_root = Path(args.repo_root).resolve()
    metadata_csv = repo_root / args.metadata_csv
    output_dir = repo_root / args.output_dir

    if args.rebuild_metadata:
        summary = build_metadata(
            tbx11k_tar=str((repo_root / args.tbx11k_tar).resolve()),
            chest_xray_tar=str((repo_root / args.chest_xray_tar).resolve()),
            extract_root=str((repo_root / args.extract_root).resolve()),
            merged_output_csv=str(metadata_csv.resolve()),
            tbx11k_output_csv=str((repo_root / 'data/processed/tbx11k_metadata_rebuilt.csv').resolve()),
            chest_output_csv=str((repo_root / 'data/processed/chest_xray_metadata.csv').resolve()),
        )
        print('metadata_summary', json.dumps(summary, indent=2))

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f'Metadata CSV not found at {metadata_csv}. Either upload data/processed/merged_metadata.csv '
            'or use --rebuild-metadata with the raw tar archives present.'
        )

    metrics, _ = train_baseline_from_metadata(
        str(metadata_csv),
        str(output_dir),
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trainable_base=args.trainable_base,
        trainable_fraction=args.trainable_fraction,
        class_weight_mode=args.class_weight,
        architecture=args.architecture,
        mild_aug=(args.augmentation == 'mild'),
    )
    print('metrics', json.dumps(metrics, indent=2))
    print('saved_to', output_dir)


if __name__ == '__main__':
    main()
