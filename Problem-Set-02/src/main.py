# =============================================================
#  main.py
#  Orchestration entry point — runs the full pipeline:
#
#    STEP 1  →  Load data            (data_loader.py)
#    STEP 2  →  EDA plot             (evaluator.py)
#    STEP 3  →  Encode features      (preprocessor.py)
#    STEP 4  →  Train / Test split   (preprocessor.py)
#    STEP 5  →  Oversample training  (preprocessor.py)
#    STEP 6  →  Scale features       (preprocessor.py)
#    STEP 7  →  Build model          (model.py)
#    STEP 8  →  Train model          (model.py)
#    STEP 9  →  Cross-validate       (model.py)
#    STEP 10 →  Predict              (main.py)
#    STEP 11 →  Evaluate + plots     (evaluator.py)
#    STEP 12 →  Save outputs         (model.py + evaluator.py)
# =============================================================

import os
import sys

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
import preprocessor
import model as mdl
import evaluator

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'bank-full.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 60)
    print("  BANK TERM DEPOSIT PREDICTION — LOGISTIC REGRESSION")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # STEP 1 : Load Data
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 : LOAD DATA")
    print("=" * 60)
    df = data_loader.load_data(DATA_PATH)
    data_loader.summarise(df)

    # ──────────────────────────────────────────────────────────
    # STEP 2 : EDA
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 : EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    evaluator.plot_eda(df, OUTPUT_DIR)
    print("\n  EDA Key Observations:")
    print("   • 88.3% No  vs  11.7% Yes  →  significant class imbalance")
    print("   • Longer call duration strongly correlates with subscription")
    print("   • Students & retired customers have highest subscription rates")
    print("   • Mar, Sep, Oct, Dec months show the highest subscription rates")
    print("   • Customers without a housing/personal loan subscribe more")
    print("   • Previous successful campaign outcome is a strong predictor")

    # ──────────────────────────────────────────────────────────
    # STEP 3 : Encode Features
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 : ENCODE FEATURES")
    print("=" * 60)
    print("\n  Strategy:")
    print("   • Binary columns (default, housing, loan, y) → 0/1 map")
    print("   • Categorical columns → one-hot encoding (drop_first=True)")
    print("     drop_first avoids the dummy variable trap (multicollinearity)")
    data_enc, feature_names = preprocessor.encode(df)

    # ──────────────────────────────────────────────────────────
    # STEP 4 : Train / Test Split
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 : TRAIN / TEST SPLIT  (80/20 Stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = preprocessor.split(data_enc, feature_names)

    # ──────────────────────────────────────────────────────────
    # STEP 5 : Random Oversampling
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 : RANDOM OVERSAMPLING  (train set only)")
    print("=" * 60)
    print("\n  Why oversample?")
    print("   Original ratio is ~8:1 (No:Yes). Without correction the")
    print("   model learns to almost always predict 'No' and appears")
    print("   88% accurate while being useless for finding subscribers.")
    print("   Oversampling duplicates minority class rows until 1:1.")
    X_train_bal, y_train_bal = preprocessor.oversample(X_train, y_train)

    # ──────────────────────────────────────────────────────────
    # STEP 6 : Feature Scaling
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 : FEATURE SCALING  (StandardScaler)")
    print("=" * 60)
    X_train_sc, X_test_sc, scaler = preprocessor.scale(X_train_bal, X_test)

    # ──────────────────────────────────────────────────────────
    # STEP 7 : Build Model
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7 : BUILD LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    print("\n  Model equation:")
    print("   P(y=1|X) = σ(β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ)")
    print("   σ(z) = 1 / (1 + e^{−z})  [sigmoid function]")
    print("   Decision: 'Yes' if P ≥ 0.5,  'No' otherwise")
    lr_model = mdl.build()

    # ──────────────────────────────────────────────────────────
    # STEP 8 : Train Model
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8 : TRAIN MODEL")
    print("=" * 60)
    lr_model = mdl.train(lr_model, X_train_sc, y_train_bal)

    # ──────────────────────────────────────────────────────────
    # STEP 9 : Cross-Validation
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 9 : CROSS-VALIDATION  (5-Fold Stratified)")
    print("=" * 60)
    cv_scores = mdl.cross_validate(lr_model, X_train_sc, y_train_bal)
    evaluator.plot_cv_scores(cv_scores, OUTPUT_DIR)

    # ──────────────────────────────────────────────────────────
    # STEP 10 : Predict on Test Set
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 10 : PREDICT ON TEST SET")
    print("=" * 60)
    y_pred = lr_model.predict(X_test_sc)
    y_prob = lr_model.predict_proba(X_test_sc)[:, 1]

    print(f"\n  Total test samples      :  {len(y_test):,}")
    print(f"  Predicted as No  (0)    :  {(y_pred == 0).sum():,}")
    print(f"  Predicted as Yes (1)    :  {(y_pred == 1).sum():,}")
    print(f"\n  Sample predictions (first 10 rows):")
    print(f"   {'Actual':<10} {'Predicted':<12} {'P(Yes)':>8}  {'Result'}")
    print("   " + "─" * 42)
    for act, pred, prob in zip(y_test.values[:10], y_pred[:10], y_prob[:10]):
        mark = '✔' if act == pred else '✗'
        print(f"   {act:<10} {pred:<12} {prob:>8.4f}  {mark}")

    # ──────────────────────────────────────────────────────────
    # STEP 11 : Evaluate — Plots + Metrics
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 11 : EVALUATE — PLOTS + METRICS")
    print("=" * 60)
    evaluator.plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR)
    auc = evaluator.plot_roc_curve(y_test, y_prob, OUTPUT_DIR)
    ap  = evaluator.plot_precision_recall(y_test, y_prob, OUTPUT_DIR)
    evaluator.plot_feature_importance(lr_model, feature_names, OUTPUT_DIR)
    metrics = evaluator.compute_and_save_metrics(
        y_test, y_pred, y_prob, cv_scores, auc, ap, OUTPUT_DIR
    )

    # ──────────────────────────────────────────────────────────
    # STEP 12 : Save Model Info
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 12 : SAVE MODEL INFO")
    print("=" * 60)
    mdl.save_model_info(lr_model, cv_scores, feature_names, OUTPUT_DIR)

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │          LOGISTIC REGRESSION — FINAL RESULTS        │
  ├─────────────────────────────────────────────────────┤
  │  Dataset         :  45,211 samples, 16 features     │
  │  Train (raw)     :  36,168  →  oversampled 1:1      │
  │  Test            :   9,043  (held-out, untouched)   │
  ├─────────────────────────────────────────────────────┤
  │  Accuracy        :  {metrics['Accuracy']*100:.2f}%                       │
  │  ROC-AUC         :  {metrics['ROC-AUC']:.4f}                      │
  │  Avg Precision   :  {metrics['Average Precision']:.4f}                      │
  │  F1-Score (Yes)  :  {metrics['F1-Score (Yes)']:.4f}                      │
  │  Sensitivity     :  {metrics['Sensitivity/Recall']*100:.1f}%                       │
  │  Specificity     :  {metrics['Specificity']*100:.1f}%                       │
  │  CV ROC-AUC      :  {metrics['CV ROC-AUC Mean']:.4f}  (±{metrics['CV ROC-AUC Std']:.4f})        │
  ├─────────────────────────────────────────────────────┤
  │  Outputs saved in  ./outputs/                       │
  │    eda_overview.png       confusion_matrix.png      │
  │    roc_curve.png          precision_recall.png      │
  │    feature_importance.png cv_scores.png             │
  │    metrics.json           model_info.json           │
  └─────────────────────────────────────────────────────┘
    """)
    print("  ✔  Pipeline complete!\n")


if __name__ == '__main__':
    main()
