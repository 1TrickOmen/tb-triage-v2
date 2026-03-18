from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, callbacks
import tensorflow as tf

from .models import build_mobilenetv2
from src.data.splits import stratified_train_val_test_split

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 2
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-4


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes'}
    return bool(value)


def _configure_tensorflow() -> None:
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def load_images_from_metadata(df: pd.DataFrame, image_size=IMAGE_SIZE):
    images, labels = [], []
    label_map = {'Normal': 0, 'TB': 1}
    for _, row in df.iterrows():
        img = cv2.imread(str(row['image_path']))
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label_map[row['label_final']])
    return np.array(images), np.array(labels)


def _resolve_metadata_paths(df: pd.DataFrame, metadata_csv: str) -> pd.DataFrame:
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


def train_baseline_from_metadata(
    metadata_csv: str,
    output_dir: str,
    *,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    trainable_base: bool = False,
    class_weight_mode: str = 'none',
):
    _configure_tensorflow()

    df = pd.read_csv(metadata_csv)
    df['include_for_training'] = df['include_for_training'].apply(_coerce_bool)
    df = df[df['include_for_training']].copy()
    df = df[df['label_final'].isin(['Normal', 'TB'])].copy()
    df = _resolve_metadata_paths(df, metadata_csv)
    df['image_exists'] = df['image_path'].apply(lambda p: Path(p).exists())
    missing_count = int((~df['image_exists']).sum())
    if missing_count:
        raise FileNotFoundError(f'{missing_count} metadata rows point to missing image files. Rebuild metadata after extracting archives.')
    if df['label_final'].nunique() < 2:
        raise ValueError('Need at least two classes with valid training rows.')

    images, labels = load_images_from_metadata(df, image_size=image_size)
    if len(images) == 0:
        raise ValueError('No images were loaded from metadata.')
    if len(set(labels.tolist())) < 2:
        raise ValueError('Loaded images do not contain both Normal and TB classes.')

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        images, labels, seed=SEED
    )

    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

    class_weight = None
    if class_weight_mode == 'balanced':
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
    elif class_weight_mode != 'none':
        raise ValueError("class_weight_mode must be 'none' or 'balanced'")

    model, last_conv_layer_name = build_mobilenetv2(
        input_shape=tuple(image_size) + (3,),
        num_classes=NUM_CLASSES,
        trainable_base=trainable_base,
    )
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-7)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'image_size': list(image_size),
        'batch_size': int(batch_size),
        'epochs_requested': int(epochs),
        'learning_rate': float(learning_rate),
        'trainable_base': bool(trainable_base),
        'class_weight_mode': class_weight_mode,
        'class_weight': class_weight,
        'tensorflow_version': tf.__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'TB'], output_dict=True),
        'roc_auc': float(roc_auc_score(y_test, y_pred_prob[:, 1])),
        'pr_auc': float(average_precision_score(y_test, y_pred_prob[:, 1])),
        'last_conv_layer_name': last_conv_layer_name,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'mobilenetv2_baseline.keras')
    with open(output / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(output / 'history.json', 'w') as f:
        json.dump(history.history, f, indent=2)

    return metrics, history.history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--trainable-base', action='store_true')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none')
    args = parser.parse_args()
    train_baseline_from_metadata(
        args.metadata_csv,
        args.output_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trainable_base=args.trainable_base,
        class_weight_mode=args.class_weight,
    )
