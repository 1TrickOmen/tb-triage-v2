from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(X, y, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError('train/val/test ratios must sum to 1.0')

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
