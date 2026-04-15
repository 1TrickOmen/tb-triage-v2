from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, callbacks
import tensorflow as tf

from .data_utils import IMAGE_SIZE, SEED, coerce_bool, load_images_from_metadata, resolve_metadata_paths
from .models import build_mobilenetv2, build_densenet121
from src.data.splits import stratified_train_val_test_split


EXPLICIT_SPLIT_COLUMN_CANDIDATES = ['experiment_split', 'split']

BATCH_SIZE = 32
NUM_CLASSES = 2
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_AUGMENTATION = 'mild'


def _select_explicit_split_column(df: pd.DataFrame) -> str | None:
    for column in EXPLICIT_SPLIT_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue
        values = set(df[column].dropna().astype(str).str.strip().str.lower())
        if {'train', 'val', 'test'}.issubset(values):
            return column
    return None


def _load_split_subset(df: pd.DataFrame, split_column: str, split_name: str, image_size):
    split_df = df[df[split_column].astype(str).str.strip().str.lower() == split_name].copy()
    if split_df.empty:
        raise ValueError(f'Metadata split column {split_column!r} does not contain any {split_name!r} rows.')
    images, labels = load_images_from_metadata(split_df, image_size=image_size)
    if len(images) == 0:
        raise ValueError(f'No images were loaded for split {split_name!r}.')
    split_df = split_df.reset_index(drop=True)
    sample_weights = None
    if 'sample_weight' in split_df.columns:
        sample_weights = split_df['sample_weight'].fillna(1.0).astype(float).to_numpy()
    return split_df, images, labels, sample_weights


def _configure_tensorflow() -> None:
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def train_baseline_from_metadata(
    metadata_csv: str,
    output_dir: str,
    *,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    trainable_base: bool = False,
    trainable_fraction: float | None = None,
    class_weight_mode: str = 'none',
    architecture: str = 'mobilenetv2',
    mild_aug: str = DEFAULT_AUGMENTATION,
):
    _configure_tensorflow()

    df = pd.read_csv(metadata_csv)
    df['include_for_training'] = df['include_for_training'].apply(coerce_bool)
    df = df[df['include_for_training']].copy()
    df = df[df['label_final'].isin(['Normal', 'TB'])].copy()
    df = resolve_metadata_paths(df, metadata_csv)
    df['image_exists'] = df['image_path'].apply(lambda p: Path(p).exists())
    missing_count = int((~df['image_exists']).sum())
    if missing_count:
        raise FileNotFoundError(f'{missing_count} metadata rows point to missing image files. Rebuild metadata after extracting archives.')
    if df['label_final'].nunique() < 2:
        raise ValueError('Need at least two classes with valid training rows.')

    split_column = _select_explicit_split_column(df)
    train_sample_weight = None
    if split_column:
        train_df, X_train, y_train, train_sample_weight = _load_split_subset(df, split_column, 'train', image_size)
        val_df, X_val, y_val, _ = _load_split_subset(df, split_column, 'val', image_size)
        test_df, X_test, y_test, _ = _load_split_subset(df, split_column, 'test', image_size)
        split_summary = {
            split_column: split_column,
            'train_rows': int(len(train_df)),
            'val_rows': int(len(val_df)),
            'test_rows': int(len(test_df)),
            'train_sources': train_df['source_dataset'].value_counts().to_dict() if 'source_dataset' in train_df.columns else {},
            'val_sources': val_df['source_dataset'].value_counts().to_dict() if 'source_dataset' in val_df.columns else {},
            'test_sources': test_df['source_dataset'].value_counts().to_dict() if 'source_dataset' in test_df.columns else {},
        }
    else:
        images, labels = load_images_from_metadata(df, image_size=image_size)
        if len(images) == 0:
            raise ValueError('No images were loaded from metadata.')
        if len(set(labels.tolist())) < 2:
            raise ValueError('Loaded images do not contain both Normal and TB classes.')

        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
            images, labels, seed=SEED
        )
        split_summary = {
            'split_column': '',
            'split_strategy': 'random_stratified_from_training_rows',
        }

    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    if mild_aug == 'none':
        datagen = ImageDataGenerator()
        augmentation_mode = 'none'
    elif mild_aug == 'mild':
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='constant',
            cval=0
        )
        augmentation_mode = 'mild'
    else:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            shear_range=15,
            fill_mode='constant',
            cval=0
        )
        augmentation_mode = 'strong'
    datagen.fit(X_train)

    train_generator = datagen.flow(X_train, y_train, sample_weight=train_sample_weight, batch_size=batch_size, shuffle=True)

    class_weight = None
    if class_weight_mode == 'balanced':
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
    elif class_weight_mode != 'none':
        raise ValueError("class_weight_mode must be 'none' or 'balanced'")

    if train_sample_weight is not None and class_weight is not None:
        raise ValueError('Use either metadata sample_weight or class_weight, not both at once.')

    model_kwargs = {
        'input_shape': tuple(image_size) + (3,),
        'num_classes': NUM_CLASSES,
        'trainable_base': trainable_base,
        'trainable_fraction': trainable_fraction,
    }
    if architecture == 'densenet121':
        model, last_conv_layer_name = build_densenet121(**model_kwargs)
    else:
        model, last_conv_layer_name = build_mobilenetv2(**model_kwargs)

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

    sample_weight_summary = None
    if train_sample_weight is not None:
        sample_weight_summary = {
            'min': float(np.min(train_sample_weight)),
            'max': float(np.max(train_sample_weight)),
            'mean': float(np.mean(train_sample_weight)),
            'sum': float(np.sum(train_sample_weight)),
        }
        if split_column and 'source_dataset' in train_df.columns and 'label_final' in train_df.columns:
            per_group = train_df[['source_dataset', 'label_final']].copy()
            per_group['sample_weight'] = train_sample_weight
            sample_weight_summary['sum_by_source_label'] = {
                f'{source_dataset}::{label_final}': float(weight_sum)
                for (source_dataset, label_final), weight_sum in per_group.groupby(['source_dataset', 'label_final'])['sample_weight'].sum().items()
            }

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
        'trainable_fraction': None if trainable_fraction is None else float(trainable_fraction),
        'class_weight_mode': class_weight_mode,
        'class_weight': class_weight,
        'architecture': architecture,
        'augmentation_mode': augmentation_mode,
        'sample_weight_column': 'sample_weight' if train_sample_weight is not None else '',
        'sample_weight_summary': sample_weight_summary,
        'split_summary': split_summary,
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
    model.save(output / f'{architecture}_baseline.keras')
    pd.DataFrame({
        'y_true': y_test.astype(int),
        'prob_normal': y_pred_prob[:, 0].astype(float),
        'prob_tb': y_pred_prob[:, 1].astype(float),
        'pred_label_default_0_5': y_pred.astype(int),
    }).to_csv(output / 'test_predictions.csv', index=False)
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
    parser.add_argument('--trainable-fraction', type=float, default=None, help='Fraction of backbone layers to unfreeze from the end, e.g. 0.25')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none')
    parser.add_argument('--architecture', choices=['mobilenetv2', 'densenet121'], default='mobilenetv2')
    parser.add_argument('--augmentation', choices=['none', 'mild', 'strong'], default='mild')
    args = parser.parse_args()
    train_baseline_from_metadata(
        args.metadata_csv,
        args.output_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trainable_base=args.trainable_base,
        trainable_fraction=args.trainable_fraction,
        class_weight_mode=args.class_weight,
        architecture=args.architecture,
        mild_aug=args.augmentation,
    )
