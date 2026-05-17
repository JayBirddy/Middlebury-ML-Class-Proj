# shared data loading pipeline used by all models.

import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.data_preprocess import (
    add_cci,
    extract_demographics,
    replace_question_marks,
    drop_all_diag_missing,
    drop_columns,
    remove_nonreadmit_patients,
    drop_invalid_demographics,
    impute_and_encode,
)

def load_data():
    print("Loading dataset...")
    raw = fetch_ucirepo(id=296)
    X   = raw.data.features.copy()
    y   = raw.data.targets.copy()
    y   = (y['readmitted'] == '<30').astype(int).values

    X = replace_question_marks(X)

    X = add_cci(X)

    demo = extract_demographics(X)

    X, y, demo = drop_all_diag_missing(X, y, demo)

    X, y, demo = drop_invalid_demographics(X, y, demo)

    X, y, demo = remove_nonreadmit_patients(X, y, demo)

    X = drop_columns(X)

    X = impute_and_encode(X)

    print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Positive class (readmitted <30d): {y.mean()*100:.1f}%")
    return X, y, demo


def split_and_scale(X, y, demo, test_size=0.2, random_state=42):
    """
    80/20 split. Scaler fitted on train only.
    """
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train   = X.iloc[idx_train]
    X_test    = X.iloc[idx_test]
    y_train   = y[idx_train]
    y_test    = y[idx_test]
    demo_test = demo.iloc[idx_test].reset_index(drop=True)

    feature_cols = list(X_train.columns)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols, demo_test