from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = (256, 256)
SEED = 42


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes'}
    return bool(value)


def load_images_from_metadata(df: pd.DataFrame, image_size=IMAGE_SIZE):
    images, labels = [], []
    label_map = {'Normal': 0, 'TB': 1}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for _, row in df.iterrows():
        img = cv2.imread(str(row['image_path']))
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        images.append(img)
        labels.append(label_map[row['label_final']])
    return np.array(images), np.array(labels)


def resolve_metadata_paths(df: pd.DataFrame, metadata_csv: str) -> pd.DataFrame:
    metadata_path = Path(metadata_csv).resolve()
    repo_root = metadata_path.parent.parent.parent

    def resolve_path(value: str) -> str:
        p = Path(value)
        if p.is_absolute():
            return str(p)
        candidate_from_cwd = Path.cwd() / p
        if candidate_from_cwd.exists():
            return str(candidate_from_cwd.resolve())
        candidate_from_repo = repo_root / p
        if candidate_from_repo.exists():
            return str(candidate_from_repo.resolve())
        return str(candidate_from_repo.resolve())

    df = df.copy()
    df['image_path'] = df['image_path'].astype(str).apply(resolve_path)
    return df
