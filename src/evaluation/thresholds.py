from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.classification.data_utils import IMAGE_SIZE, SEED, coerce_bool, load_images_from_metadata, resolve_metadata_paths
from src.data.splits import stratified_train_val_test_split


DEFAULT_THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95,
]


def load_test_split_from_metadata(metadata_csv: str, image_size=IMAGE_SIZE, seed: int = SEED):
    df = pd.read_csv(metadata_csv)
    df['include_for_training'] = df['include_for_training'].apply(coerce_bool)
    df = df[df['include_for_training']].copy()
    df = df[df['label_final'].isin(['Normal', 'TB'])].copy()
    df = resolve_metadata_paths(df, metadata_csv)
    df['image_exists'] = df['image_path'].apply(lambda p: Path(p).exists())
    missing_count = int((~df['image_exists']).sum())
    if missing_count:
        raise FileNotFoundError(f'{missing_count} metadata rows point to missing image files. Rebuild metadata after extracting archives.')

    images, labels = load_images_from_metadata(df, image_size=image_size)
    _, _, X_test, _, _, y_test = stratified_train_val_test_split(images, labels, seed=seed)
    X_test = X_test.astype('float32') / 255.0
    return X_test, y_test


def predict_test_probabilities(model_path: str, metadata_csv: str, image_size=IMAGE_SIZE, seed: int = SEED) -> pd.DataFrame:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    X_test, y_test = load_test_split_from_metadata(metadata_csv, image_size=image_size, seed=seed)
    y_pred_prob = model.predict(X_test, verbose=0)
    return pd.DataFrame({
        'y_true': y_test.astype(int),
        'prob_normal': y_pred_prob[:, 0].astype(float),
        'prob_tb': y_pred_prob[:, 1].astype(float),
    })


def evaluate_thresholds(predictions_df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    y_true = predictions_df['y_true'].astype(int).to_numpy()
    prob_tb = predictions_df['prob_tb'].astype(float).to_numpy()

    for threshold in thresholds:
        y_pred = (prob_tb >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            'threshold': float(threshold),
            'precision_tb': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall_tb': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_tb': float(f1_score(y_true, y_pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0,
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) else 0.0,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'predicted_tb': int(y_pred.sum()),
            'predicted_normal': int((1 - y_pred).sum()),
        })

    return pd.DataFrame(rows).sort_values('threshold').reset_index(drop=True)


def select_threshold_for_target_recall(threshold_metrics: pd.DataFrame, target_recall: float) -> dict | None:
    eligible = threshold_metrics[threshold_metrics['recall_tb'] >= float(target_recall)].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        by=['precision_tb', 'f1_tb', 'threshold'],
        ascending=[False, False, False],
    )
    return eligible.iloc[0].to_dict()
