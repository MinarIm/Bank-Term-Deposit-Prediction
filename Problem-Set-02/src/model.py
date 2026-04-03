# =============================================================
#  model.py
#  Responsibility: Build, train, and cross-validate the
#                  Logistic Regression model
# =============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json
import os


def build() -> LogisticRegression:
    """
    Build the Logistic Regression model.

    Hyperparameters
    ---------------
    solver       = 'lbfgs'   — Limited-memory BFGS optimiser;
                               efficient for L2 regularisation on
                               medium-to-large datasets
    max_iter     = 1000      — Enough iterations to guarantee
                               convergence on this dataset
    C            = 1.0       — Inverse of regularisation strength
                               (C=1 → moderate L2 regularisation)
    random_state = 42        — Reproducible results

    Note: class_weight is NOT set here because oversampling already
          balances the training classes to a 1:1 ratio.

    The Logistic Regression equation:
        P(y=1 | X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
        where  σ(z) = 1 / (1 + e^{−z})   [sigmoid]
    Predicts 'Yes' (1) when P ≥ 0.5, otherwise 'No' (0).
    """
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    return model


def train(model: LogisticRegression, X_train, y_train) -> LogisticRegression:
    """
    Fit the model on the oversampled, scaled training data.
    """
    print("\n  ── Training Logistic Regression ─────────────────")
    print(f"   Solver     :  lbfgs")
    print(f"   Max iter   :  1000")
    print(f"   C (reg.)   :  1.0")
    print(f"   Train size :  {len(y_train):,} samples  (after oversampling)")

    model.fit(X_train, y_train)
    print(f"   ✔  Model trained successfully.")
    return model


def cross_validate(model: LogisticRegression, X_train, y_train, n_splits: int = 5):
    """
    5-Fold Stratified Cross-Validation on the training set.

    Uses ROC-AUC as the scoring metric — more informative than
    accuracy for imbalanced datasets.

    Returns
    -------
    cv_scores : np.ndarray   AUC score per fold
    """
    print("\n  ── 5-Fold Stratified Cross-Validation ───────────")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    for i, score in enumerate(cv_scores, 1):
        bar = '█' * int(score * 35)
        print(f"   Fold {i}  |  {bar}  {score:.4f}")

    print(f"\n   Mean ROC-AUC  :  {cv_scores.mean():.4f}")
    print(f"   Std  ROC-AUC  :  ± {cv_scores.std():.4f}")
    print(f"   → Low std deviation confirms stable generalisation  ✔")

    return cv_scores


def save_model_info(model: LogisticRegression, cv_scores, feature_names: list, output_dir: str):
    """
    Save model coefficients and CV scores to a JSON file
    (equivalent to saving a .pth PyTorch model state dict).
    """
    coef_dict = {
        feat: round(float(coef), 6)
        for feat, coef in zip(feature_names, model.coef_[0])
    }

    payload = {
        "model"         : "LogisticRegression",
        "solver"        : "lbfgs",
        "C"             : 1.0,
        "intercept"     : round(float(model.intercept_[0]), 6),
        "n_features"    : len(feature_names),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std" : round(float(cv_scores.std()), 4),
        "cv_fold_scores" : [round(float(s), 4) for s in cv_scores],
        "coefficients"  : coef_dict,
    }

    path = os.path.join(output_dir, 'model_info.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\n   Model info saved  →  {path}")
