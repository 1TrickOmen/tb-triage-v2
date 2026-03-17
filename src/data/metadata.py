from pathlib import Path
import pandas as pd


def load_metadata_csv(metadata_file: str) -> pd.DataFrame:
    return pd.read_csv(metadata_file)


def standardize_binary_tb_labels(df: pd.DataFrame, raw_label_col: str = 'ptb') -> pd.DataFrame:
    out = df.copy()
    out[raw_label_col] = pd.to_numeric(out[raw_label_col], errors='coerce')
    out = out.dropna(subset=[raw_label_col])
    out[raw_label_col] = out[raw_label_col].astype(int)
    out['label_final'] = out[raw_label_col].map({1: 'TB', 0: 'Normal'})
    return out


def attach_image_paths(df: pd.DataFrame, image_dir: str, id_col: str = 'id') -> pd.DataFrame:
    out = df.copy()
    out['image_id'] = out[id_col].astype(str) + '.png'
    out['image_path'] = out['image_id'].apply(lambda x: str(Path(image_dir) / x))
    out['image_exists'] = out['image_path'].apply(lambda p: Path(p).exists())
    return out
