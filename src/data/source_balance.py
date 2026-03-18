from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.classification.data_utils import coerce_bool

LABELS = ['Normal', 'TB']
DEFAULT_WEIGHT_COLUMN = 'sample_weight'


def _normalize_group_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.items()}


def _nested_counts(df: pd.DataFrame, first: str, second: str) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    grouped = df.groupby([first, second]).size()
    for (outer, inner), value in grouped.items():
        nested.setdefault(str(outer), {})[str(inner)] = int(value)
    return nested


def add_source_balanced_sample_weights(
    metadata_csv: str,
    output_metadata_csv: str,
    *,
    split_column: str = 'experiment_split',
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    balance_mode: str = 'source_label',
    normal_weight_multiplier: float = 1.0,
) -> dict:
    if balance_mode not in {'source_label', 'normal_by_source'}:
        raise ValueError("balance_mode must be 'source_label' or 'normal_by_source'")
    if normal_weight_multiplier <= 0:
        raise ValueError('normal_weight_multiplier must be > 0')

    df = pd.read_csv(metadata_csv)
    if split_column not in df.columns:
        raise ValueError(f'Metadata is missing split column {split_column!r}. Prepare explicit train/val/test splits first.')
    if 'label_final' not in df.columns or 'source_dataset' not in df.columns:
        raise ValueError('Metadata must contain label_final and source_dataset columns.')

    if 'include_for_training' in df.columns:
        include = df['include_for_training'].apply(coerce_bool)
    else:
        include = pd.Series([True] * len(df), index=df.index)

    split_series = df[split_column].astype(str).str.strip().str.lower()
    label_series = df['label_final'].astype(str)
    train_mask = include & (split_series == 'train') & label_series.isin(LABELS)
    train_df = df.loc[train_mask].copy()
    if train_df.empty:
        raise ValueError('No eligible training rows found for weighting.')

    train_df[weight_column] = 1.0

    if balance_mode == 'source_label':
        group_sizes = train_df.groupby(['source_dataset', 'label_final']).size()
        if group_sizes.empty:
            raise ValueError('Could not compute source-label group sizes.')
        target_group_mass = len(train_df) / float(len(group_sizes))
        for (source_dataset, label_final), group_size in group_sizes.items():
            weight = target_group_mass / float(group_size)
            if label_final == 'Normal':
                weight *= float(normal_weight_multiplier)
            mask = (
                train_df['source_dataset'].astype(str) == str(source_dataset)
            ) & (
                train_df['label_final'].astype(str) == str(label_final)
            )
            train_df.loc[mask, weight_column] = float(weight)
    else:
        normal_mask = train_df['label_final'].astype(str) == 'Normal'
        normal_counts = train_df.loc[normal_mask].groupby('source_dataset').size()
        if normal_counts.empty:
            raise ValueError('No Normal training rows found for normal_by_source weighting.')
        target_normal_mass = normal_mask.sum() / float(len(normal_counts))
        for source_dataset, group_size in normal_counts.items():
            weight = (target_normal_mass / float(group_size)) * float(normal_weight_multiplier)
            mask = normal_mask & (train_df['source_dataset'].astype(str) == str(source_dataset))
            train_df.loc[mask, weight_column] = float(weight)

    df = df.copy()
    df[weight_column] = 1.0
    df.loc[train_df.index, weight_column] = train_df[weight_column].astype(float)

    output_path = Path(output_metadata_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    weighted_group_means = train_df.groupby(['source_dataset', 'label_final'])[weight_column].mean()
    weighted_group_mass = train_df.groupby(['source_dataset', 'label_final'])[weight_column].sum()

    total_weight_by_source_label: dict[str, dict[str, float]] = {}
    for (source_dataset, label_final), weight_sum in weighted_group_mass.items():
        total_weight_by_source_label.setdefault(str(source_dataset), {})[str(label_final)] = float(weight_sum)

    return {
        'input_metadata_csv': str(Path(metadata_csv).resolve()),
        'output_metadata_csv': str(output_path.resolve()),
        'split_column': split_column,
        'weight_column': weight_column,
        'balance_mode': balance_mode,
        'normal_weight_multiplier': float(normal_weight_multiplier),
        'train_rows_weighted': int(len(train_df)),
        'train_counts_by_source_label': _nested_counts(train_df, 'source_dataset', 'label_final'),
        'train_total_weight_by_source_label': total_weight_by_source_label,
        'train_total_weight_by_group': {
            f'{source_dataset}::{label_final}': float(weight_sum)
            for (source_dataset, label_final), weight_sum in weighted_group_mass.items()
        },
        'train_mean_weight_by_group': {
            f'{source_dataset}::{label_final}': float(weight_mean)
            for (source_dataset, label_final), weight_mean in weighted_group_means.items()
        },
        'train_counts_by_label': _normalize_group_counts(train_df['label_final'].value_counts()),
        'train_counts_by_source': _normalize_group_counts(train_df['source_dataset'].value_counts()),
    }


def write_summary_json(summary: dict, output_json: str) -> None:
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(summary, f, indent=2, default=str)
