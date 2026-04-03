# =============================================================
#  preprocessor.py
#  Responsibility: Full preprocessing pipeline
#    Step 1 - Encode target & categorical features (one-hot)
#    Step 2 - Train / Test split (stratified 80/20)
#    Step 3 - Random oversampling on training set only
#    Step 4 - Feature scaling (StandardScaler)
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def encode(df: pd.DataFrame):
    """
    Step 1 — Encoding
    -----------------
    • Binary columns (default, housing, loan, y) → 0 / 1 integer map
    • Multi-class categorical columns → one-hot encoded via pd.get_dummies
      with drop_first=True to avoid the dummy variable trap (multicollinearity)

    Returns
    -------
    data : pd.DataFrame   fully encoded dataframe
    feature_names : list  column names of X after encoding
    """
    print("\n  ── Step 1 : Encoding ────────────────────────────")

    data = df.copy()

    # Binary columns: direct integer mapping
    binary_map = {'yes': 1, 'no': 0}
    for col in ['default', 'housing', 'loan', 'y']:
        data[col] = data[col].map(binary_map)
        print(f"   {col:<14}  binary map  →  no=0, yes=1")

    # Multi-class categorical columns: one-hot encoding
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    print(f"\n   One-hot encoding  →  {cat_cols}")
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    print(f"   Shape after encoding  :  {data.shape}")

    feature_names = [c for c in data.columns if c != 'y']
    print(f"   Total features        :  {len(feature_names)}")
    return data, feature_names


def split(data: pd.DataFrame, feature_names: list, test_size: float = 0.20, random_state: int = 42):
    """
    Step 2 — Train / Test Split
    ---------------------------
    • 80% train, 20% test
    • stratify=y ensures both sets preserve the original 88:12 class ratio
    """
    print("\n  ── Step 2 : Train / Test Split ──────────────────")

    X = data[feature_names]
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"   Split ratio   :  80% train  /  20% test")
    print(f"   Training set  :  {len(X_train):,} samples")
    print(f"   Test set      :  {len(X_test):,} samples")
    print(f"   Train class balance  →  No: {(y_train==0).sum():,}  |  Yes: {(y_train==1).sum():,}")
    print(f"   Test  class balance  →  No: {(y_test==0).sum():,}  |  Yes: {(y_test==1).sum():,}")

    return X_train, X_test, y_train, y_test


def oversample(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Step 3 — Random Oversampling (training set ONLY)
    -------------------------------------------------
    • Minority class (y=1, subscribed) is randomly duplicated until
      it matches the majority class count → 1:1 balanced ratio
    • Applied ONLY on training data → no information leakage from test set
    """
    print("\n  ── Step 3 : Random Oversampling (train only) ────")

    train = pd.concat([X_train, y_train], axis=1)
    majority = train[train['y'] == 0]
    minority = train[train['y'] == 1]

    print(f"   Before oversampling  →  Majority (No): {len(majority):,}  |  Minority (Yes): {len(minority):,}")

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=random_state
    )

    train_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=random_state)

    X_train_bal = train_balanced.drop('y', axis=1)
    y_train_bal = train_balanced['y']

    print(f"   After  oversampling  →  Majority (No): {(y_train_bal==0).sum():,}  |  Minority (Yes): {(y_train_bal==1).sum():,}")
    print(f"   New training size    :  {len(X_train_bal):,} samples  (1:1 ratio)")

    return X_train_bal, y_train_bal


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Step 4 — Feature Scaling (StandardScaler)
    ------------------------------------------
    • Transforms each feature to zero mean and unit variance
    • fit_transform on train → transform ONLY on test (prevents data leakage)
    • Essential for gradient-based optimisation and fair coefficient comparison
    """
    print("\n  ── Step 4 : Feature Scaling (StandardScaler) ────")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"   Formula   :  z = (x − μ) / σ   →   mean≈0, std≈1")
    print(f"   Fit on training data only  →  no data leakage into test set  ✔")

    return X_train_sc, X_test_sc, scaler
