from pathlib import Path
import pandas as pd

from .metadata import load_metadata_csv, standardize_binary_tb_labels, attach_image_paths


def ingest_kaggle_tb_dataset(metadata_file: str, image_dir: str, source_dataset: str) -> pd.DataFrame:
    df = load_metadata_csv(metadata_file)
    df = standardize_binary_tb_labels(df, raw_label_col='ptb')
    df = attach_image_paths(df, image_dir=image_dir, id_col='id')
    df = df[df['image_exists']].copy()
    df['source_dataset'] = source_dataset
    df['patient_id'] = df.get('id', pd.Series([None] * len(df)))
    df['mask_path'] = None
    df['view_position'] = df['position'] if 'position' in df.columns else None
    df['split'] = None
    df['is_external_test'] = False
    df['include_for_training'] = True
    keep_cols = [
        'image_id', 'patient_id', 'source_dataset', 'ptb', 'label_final',
        'image_path', 'mask_path', 'view_position', 'split', 'is_external_test',
        'include_for_training'
    ]
    out = df[keep_cols].rename(columns={'ptb': 'label_raw'})
    out['notes'] = ''
    return out


def save_metadata(df: pd.DataFrame, output_csv: str) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
