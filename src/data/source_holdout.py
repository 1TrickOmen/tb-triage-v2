from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classification.data_utils import SEED, coerce_bool, resolve_metadata_paths

LABELS = ['Normal', 'TB']


def _counts_to_nested_dict(series: pd.Series) -> dict:
    nested: dict[str, dict[str, int]] = {}
    for key, value in series.items():
        if isinstance(key, tuple) and len(key) == 2:
            outer, inner = key
            nested.setdefault(str(outer), {})[str(inner)] = int(value)
        else:
            nested[str(key)] = int(value)
    return nested


def _load_base_metadata(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    df['include_for_training'] = df['include_for_training'].apply(coerce_bool)
    df = df[df['label_final'].isin(LABELS)].copy()
    df = resolve_metadata_paths(df, metadata_csv)
    return df.reset_index(drop=True)


def _stratified_train_val_test_indices(labels, seed: int = SEED, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError('train/val/test ratios must sum to 1.0')
    all_idx = list(range(len(labels)))
    train_idx, temp_idx = train_test_split(all_idx, test_size=(1 - train_ratio), random_state=seed, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=relative_test_ratio, random_state=seed, stratify=temp_labels)
    return train_idx, val_idx, test_idx


def make_source_holdout_metadata(metadata_csv: str, holdout_source: str, output_metadata_csv: str, output_holdout_csv: str | None = None, seed: int = SEED) -> dict:
    df = _load_base_metadata(metadata_csv)
    seen_df = df[(df['include_for_training']) & (df['source_dataset'] != holdout_source)].copy()
    holdout_df = df[(df['include_for_training']) & (df['source_dataset'] == holdout_source)].copy()

    if seen_df.empty:
        raise ValueError(f'No seen-source training rows remain after holding out {holdout_source!r}.')
    if holdout_df.empty:
        raise ValueError(f'No holdout rows found for source {holdout_source!r}.')
    if seen_df['label_final'].nunique() < 2:
        raise ValueError(f'Seen-source rows for holdout {holdout_source!r} do not contain both Normal and TB.')
    if holdout_df['label_final'].nunique() < 2:
        raise ValueError(f'Holdout rows for source {holdout_source!r} do not contain both Normal and TB.')

    train_idx, val_idx, test_idx = _stratified_train_val_test_indices(seen_df['label_final'].tolist(), seed=seed)
    seen_df = seen_df.reset_index(drop=True)
    seen_df['experiment_split'] = ''
    seen_df.loc[train_idx, 'experiment_split'] = 'train'
    seen_df.loc[val_idx, 'experiment_split'] = 'val'
    seen_df.loc[test_idx, 'experiment_split'] = 'test'
    seen_df['experiment_role'] = 'seen_source'
    seen_df['held_out_source'] = holdout_source
    seen_df['include_for_training'] = seen_df['experiment_split'].isin(['train', 'val', 'test'])
    seen_df['is_external_test'] = False

    holdout_df = holdout_df.reset_index(drop=True)
    holdout_df['experiment_split'] = 'external_eval'
    holdout_df['experiment_role'] = 'held_out_source'
    holdout_df['held_out_source'] = holdout_source
    holdout_df['include_for_training'] = False
    holdout_df['is_external_test'] = True

    experiment_df = pd.concat([seen_df, holdout_df], ignore_index=True)
    output_path = Path(output_metadata_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_df.to_csv(output_path, index=False)

    holdout_output_path = None
    if output_holdout_csv:
        holdout_output_path = Path(output_holdout_csv)
        holdout_output_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_df.to_csv(holdout_output_path, index=False)

    summary = {
        'holdout_source': holdout_source,
        'seed': int(seed),
        'output_metadata_csv': str(output_path),
        'output_holdout_csv': str(holdout_output_path) if holdout_output_path else '',
        'seen_source_counts': _counts_to_nested_dict(seen_df.groupby(['source_dataset', 'label_final']).size()),
        'seen_split_counts': _counts_to_nested_dict(seen_df.groupby(['experiment_split', 'label_final']).size()),
        'holdout_counts': _counts_to_nested_dict(holdout_df.groupby(['source_dataset', 'label_final']).size()),
        'seen_total': int(len(seen_df)),
        'holdout_total': int(len(holdout_df)),
    }
    return summary


def write_summary_json(summary: dict, output_json: str) -> None:
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(summary, f, indent=2, default=str)
