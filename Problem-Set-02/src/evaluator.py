# =============================================================
#  evaluator.py
#  Responsibility: Compute all evaluation metrics and generate
#                  every output plot + metrics.json
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score
)

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
})
BLUE  = '#1565C0'
RED   = '#C62828'
GREEN = '#2E7D32'


# =============================================================
# 1. EDA OVERVIEW  (6-panel)
# =============================================================
def plot_eda(df_raw: pd.DataFrame, output_dir: str) -> None:
    """
    6-panel EDA visualisation saved as eda_overview.png
      Panel 1 — Target class distribution (donut)
      Panel 2 — Age distribution by outcome (KDE)
      Panel 3 — Call duration by outcome (KDE)
      Panel 4 — Subscription rate by job type (bar)
      Panel 5 — Subscription rate by month (line)
      Panel 6 — Numerical correlation heatmap
    """
    print("\n  Generating eda_overview.png …")

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('EDA — Bank Marketing Dataset', fontsize=17,
                 fontweight='bold', y=1.01)

    vc = df_raw['y'].value_counts()

    # ── Panel 1 : Donut chart ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        vc.values, labels=vc.index, autopct='%1.1f%%',
        colors=[BLUE, RED], startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2.5)
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight('bold')
    ax1.set_title('Target Distribution  (y)', fontweight='bold')

    # ── Panel 2 : Age KDE ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for label, color in zip(['no', 'yes'], [BLUE, RED]):
        df_raw[df_raw['y'] == label]['age'].plot(
            kind='kde', ax=ax2, color=color, lw=2, label=label)
    ax2.set_title('Age Distribution by Outcome', fontweight='bold')
    ax2.set_xlabel('Age'); ax2.legend()

    # ── Panel 3 : Duration KDE ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for label, color in zip(['no', 'yes'], [BLUE, RED]):
        df_raw[df_raw['y'] == label]['duration'].clip(upper=1500).plot(
            kind='kde', ax=ax3, color=color, lw=2, label=label)
    ax3.set_title('Call Duration by Outcome', fontweight='bold')
    ax3.set_xlabel('Duration (seconds)'); ax3.legend()

    # ── Panel 4 : Job subscription rate ─────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    job_rate = (df_raw.groupby('job')['y']
                .apply(lambda x: (x == 'yes').mean() * 100)
                .sort_values(ascending=False))
    palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(job_rate)))
    bars = ax4.bar(job_rate.index, job_rate.values, color=palette, edgecolor='white')
    ax4.set_title('Subscription Rate by Job (%)', fontweight='bold')
    ax4.set_ylabel('Rate (%)'); ax4.set_xticklabels(job_rate.index, rotation=40, ha='right', fontsize=8)
    for bar, val in zip(bars, job_rate.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', fontsize=7.5)

    # ── Panel 5 : Monthly subscription rate ─────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    month_rate  = (df_raw.groupby('month')['y']
                   .apply(lambda x: (x == 'yes').mean() * 100)
                   .reindex(month_order).dropna())
    ax5.plot(range(len(month_rate)), month_rate.values, 'o-',
             color=BLUE, lw=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax5.fill_between(range(len(month_rate)), month_rate.values, alpha=0.12, color=BLUE)
    ax5.set_xticks(range(len(month_rate)))
    ax5.set_xticklabels(month_rate.index, rotation=30, fontsize=9)
    ax5.set_title('Subscription Rate by Month (%)', fontweight='bold')
    ax5.set_ylabel('Rate (%)')
    for i, val in enumerate(month_rate.values):
        ax5.annotate(f'{val:.1f}%', (i, val), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=8)

    # ── Panel 6 : Correlation heatmap ───────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    num_cols = ['age','balance','day','duration','campaign','pdays','previous']
    corr = df_raw[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax6, linewidths=0.5, square=True,
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax6.set_title('Numerical Correlation Heatmap', fontweight='bold')
    ax6.tick_params(axis='x', rotation=30)

    plt.savefig(os.path.join(output_dir, 'eda_overview.png'), bbox_inches='tight')
    plt.close()
    print("   ✔  Saved: eda_overview.png")


# =============================================================
# 2. CONFUSION MATRIX
# =============================================================
def plot_confusion_matrix(y_test, y_pred, output_dir: str) -> None:
    print("\n  Generating confusion_matrix.png …")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No (0)', 'Yes (1)'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    print("   ✔  Saved: confusion_matrix.png")


# =============================================================
# 3. ROC CURVE
# =============================================================
def plot_roc_curve(y_test, y_prob, output_dir: str) -> float:
    print("\n  Generating roc_curve.png …")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'ROC-AUC = {auc:.4f}')
    ax.plot([0,1],[0,1], '--', color='grey', lw=1, alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)

    # Mark optimal threshold (Youden's J)
    j_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[j_idx], tpr[j_idx], color=RED, s=100, zorder=5,
               label=f'Optimal threshold = {thresholds[j_idx]:.2f}')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontweight='bold', fontsize=13, pad=10)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), bbox_inches='tight')
    plt.close()
    print(f"   ✔  Saved: roc_curve.png   |   AUC = {auc:.4f}")
    return auc


# =============================================================
# 4. PRECISION-RECALL CURVE
# =============================================================
def plot_precision_recall(y_test, y_prob, output_dir: str) -> float:
    print("\n  Generating precision_recall.png …")
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color='#FF6F00', lw=2.5, label=f'Avg Precision = {ap:.4f}')
    ax.axhline(baseline, linestyle='--', color='grey', lw=1,
               label=f'Baseline (random) = {baseline:.2f}')
    ax.fill_between(rec, prec, alpha=0.12, color='#FF6F00')

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=13, pad=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'), bbox_inches='tight')
    plt.close()
    print(f"   ✔  Saved: precision_recall.png   |   AP = {ap:.4f}")
    return ap


# =============================================================
# 5. FEATURE IMPORTANCE  (top 20)
# =============================================================
def plot_feature_importance(model, feature_names: list, output_dir: str) -> None:
    print("\n  Generating feature_importance.png …")

    coef_df = pd.DataFrame({
        'Feature'    : feature_names,
        'Coefficient': model.coef_[0]
    }).reindex(pd.Series(model.coef_[0]).abs().sort_values(ascending=False).index)
    coef_df = coef_df.head(20).sort_values('Coefficient')

    colors = [RED if c > 0 else BLUE for c in coef_df['Coefficient']]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'],
                   color=colors, edgecolor='white', linewidth=0.8, height=0.75)
    ax.axvline(0, color='black', linewidth=0.9)

    for bar, val in zip(bars, coef_df['Coefficient']):
        x_pos = val + 0.005 if val >= 0 else val - 0.005
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=8.5)

    legend_patches = [
        plt.Rectangle((0,0),1,1, color=RED,  label='Positive  →  increases P(subscribe)'),
        plt.Rectangle((0,0),1,1, color=BLUE, label='Negative  →  decreases P(subscribe)')
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    ax.set_xlabel('Logistic Regression Coefficient', fontsize=11)
    ax.set_title('Feature Importance — Top 20 Coefficients', fontweight='bold', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()
    print("   ✔  Saved: feature_importance.png")


# =============================================================
# 6. CROSS-VALIDATION PLOT
# =============================================================
def plot_cv_scores(cv_scores, output_dir: str) -> None:
    print("\n  Generating cv_scores.png …")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [BLUE if s >= cv_scores.mean() else '#90CAF9' for s in cv_scores]
    ax.bar(range(1, len(cv_scores)+1), cv_scores, color=colors,
           edgecolor='white', linewidth=1.5)
    ax.axhline(cv_scores.mean(), color=RED, linestyle='--', lw=2,
               label=f'Mean AUC = {cv_scores.mean():.4f}')
    ax.set_xticks(range(1, len(cv_scores)+1))
    ax.set_xticklabels([f'Fold {i}' for i in range(1, len(cv_scores)+1)])
    ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.80, 0.96)
    ax.set_title('5-Fold Cross-Validation — ROC-AUC Scores', fontweight='bold', fontsize=13)
    ax.legend()
    for i, score in enumerate(cv_scores, 1):
        ax.text(i, score + 0.002, f'{score:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_scores.png'), bbox_inches='tight')
    plt.close()
    print("   ✔  Saved: cv_scores.png")


# =============================================================
# 7. COMPUTE METRICS + PRINT + SAVE metrics.json
# =============================================================
def compute_and_save_metrics(y_test, y_pred, y_prob, cv_scores,
                              auc: float, ap: float, output_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  EVALUATION METRICS")
    print("=" * 60)

    cm             = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc            = accuracy_score(y_test, y_pred)
    f1             = f1_score(y_test, y_pred)
    sensitivity    = tp / (tp + fn)
    specificity    = tn / (tn + fp)
    precision_val  = tp / (tp + fp)

    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                 No     Yes")
    print(f"  Actual  No   [{tn:>5}  {fp:>5}]")
    print(f"  Actual  Yes  [{fn:>5}  {tp:>5}]")
    print(f"\n  True  Negatives (TN) : {tn:,}")
    print(f"  False Positives (FP) : {fp:,}")
    print(f"  False Negatives (FN) : {fn:,}")
    print(f"  True  Positives (TP) : {tp:,}")

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print("  " + "-" * 37)
    metrics = {
        "Accuracy"           : round(acc, 4),
        "ROC-AUC"            : round(auc, 4),
        "Average Precision"  : round(ap, 4),
        "F1-Score (Yes)"     : round(f1, 4),
        "Sensitivity/Recall" : round(sensitivity, 4),
        "Specificity"        : round(specificity, 4),
        "Precision (Yes)"    : round(precision_val, 4),
        "CV ROC-AUC Mean"    : round(float(cv_scores.mean()), 4),
        "CV ROC-AUC Std"     : round(float(cv_scores.std()), 4),
        "True Negatives"     : int(tn),
        "False Positives"    : int(fp),
        "False Negatives"    : int(fn),
        "True Positives"     : int(tp),
    }
    for k, v in metrics.items():
        print(f"  {k:<25} {str(v):>10}")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No (0)', 'Yes (1)']))

    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✔  Metrics saved  →  {path}")

    return metrics
