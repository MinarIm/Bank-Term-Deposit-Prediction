# 🏦 Bank Term Deposit Prediction

### Logistic Regression with Modular Pipeline

---

## Problem Statement

A banking institution wants to predict whether a customer will **subscribe to a term deposit** based on their demographic profile, account details, and past campaign interactions. This is a binary classification problem where the target variable `y` represents whether a customer subscribed (`yes`) or not (`no`).

---

## Dataset

**Source:** UCI Bank Marketing Dataset (`bank-full.csv`)

| Property | Value |
|----------|-------|
| Total Records | 45,211 |
| Features | 16 input features + 1 target |
| Target Classes | `yes` (subscribed), `no` (not subscribed) |
| Class Imbalance | ~8:1 (no : yes) |
| Separator | Semicolon (`;`) |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Customer age |
| job | Categorical | Type of employment (12 categories) |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Binary | Has credit in default? |
| balance | Numerical | Average yearly account balance (€) |
| housing | Binary | Has housing loan? |
| loan | Binary | Has personal loan? |
| contact | Categorical | Contact communication type |
| day | Numerical | Last contact day of month |
| month | Categorical | Last contact month |
| duration | Numerical | Last contact duration (seconds) |
| campaign | Numerical | Number of contacts in current campaign |
| pdays | Numerical | Days since last contact (−1 = not contacted) |
| previous | Numerical | Number of contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign |

---

## Project Structure

```
Problem-Set-02/
├── data/
│   └── bank-full.csv
├── src/
│   ├── data_loader.py       # Dataset loading and full summary
│   ├── preprocessor.py      # Encoding, splitting, oversampling, scaling
│   ├── model.py             # Logistic Regression build, train, cross-validate
│   ├── evaluator.py         # All metrics computation and all visualisations
│   └── main.py              # Orchestration entry point (12 steps)
├── outputs/
│   ├── eda_overview.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall.png
│   ├── feature_importance.png
│   ├── cv_scores.png
│   ├── metrics.json
│   └── model_info.json
├── requirements.txt
└── README.md
```

---

## Methodology

### Step 1 — Load Data (`data_loader.py`)

Loads `bank-full.csv` using `pd.read_csv` with semicolon separator and prints:

- Shape, first 5 rows, data types
- Missing values check → **0 missing values**
- Numerical statistics (mean, std, min, max)
- Target class distribution → **88.3% No / 11.7% Yes**
- Unique values per categorical column

### Step 2 — Exploratory Data Analysis (`evaluator.py`)

6-panel visual saved as `eda_overview.png`:

- Target class distribution (donut chart)
- Age and call duration distributions by outcome (KDE)
- Subscription rate by job type
- Subscription rate by month (line chart)
- Numerical correlation heatmap

**Key EDA Findings:**

- **Class imbalance:** 88.3% No vs 11.7% Yes
- **Call duration** is the strongest indicator — longer calls = more likely to subscribe
- **Students & retired** customers have the highest subscription rates
- **Mar, Sep, Oct, Dec** months show notably higher subscription rates
- Customers **without** a housing or personal loan are more likely to subscribe
- A **successful previous campaign** is a strong positive predictor

### Step 3 — Encode Features (`preprocessor.py`)

**Binary columns** (`default`, `housing`, `loan`, `y`) → integer map: `no=0, yes=1`

**Multi-class categorical columns** → one-hot encoding via `pd.get_dummies` with `drop_first=True`:

- Columns encoded: `job`, `marital`, `education`, `contact`, `month`, `poutcome`
- `drop_first=True` removes one dummy per group to **avoid the dummy variable trap** (multicollinearity)
- Feature count after encoding: **42 features**

### Step 4 — Train / Test Split (`preprocessor.py`)

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

- **80%** training (36,168 samples) / **20%** test (9,043 samples)
- `stratify=y` ensures both sets preserve the original 88:12 class ratio

### Step 5 — Random Oversampling (`preprocessor.py`)

```
Before → No: 31,937  |  Yes:  4,231   (~8:1 ratio)
After  → No: 31,937  |  Yes: 31,937   ( 1:1 ratio)
Training size after oversampling: 63,874 samples
```

- Applied **on training data only** — test set is never touched → no data leakage
- Minority class (subscribed=yes) rows randomly duplicated until equal count
- Creates a balanced 1:1 training distribution so the model gives equal attention to both classes

### Step 6 — Feature Scaling (`preprocessor.py`)

```
StandardScaler: z = (x − μ) / σ   →   mean ≈ 0, std ≈ 1
```

- `fit_transform` on training data; `transform` only on test → **no data leakage**
- Essential for gradient-based optimisation to converge correctly
- Ensures all features contribute fairly regardless of their original scale

### Step 7 — Build Model (`model.py`)

**Model equation:**

```
P(y=1 | X) = σ(β₀ + β₁x₁ + β₂x₂ + … + β₄₂x₄₂)
where  σ(z) = 1 / (1 + e^{−z})   [sigmoid function]
Decision: predict 'Yes' (1) if P ≥ 0.5, else 'No' (0)
```

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solver` | `lbfgs` | Efficient L-BFGS optimiser for L2 regularisation |
| `max_iter` | `1000` | Guarantees convergence on this dataset |
| `C` | `1.0` | Moderate L2 regularisation (inverse of λ) |
| `random_state` | `42` | Reproducibility |

> **Note:** `class_weight` is not set because oversampling already balances the training data to a 1:1 ratio.

### Step 8 — Train Model (`model.py`)

```python
model.fit(X_train_sc, y_train_balanced)
# Trained on 63,874 samples (oversampled, scaled)
```

### Step 9 — Cross-Validation (`model.py`)

```
5-Fold Stratified Cross-Validation:
  Fold 1  →  0.9094
  Fold 2  →  0.9128
  Fold 3  →  0.9107
  Fold 4  →  0.9111
  Fold 5  →  0.9072

  Mean ROC-AUC  :  0.9102
  Std  ROC-AUC  :  ± 0.0019  ← very stable
```

### Step 10 — Predict on Test Set (`main.py`)

```python
y_pred = model.predict(X_test_sc)              # class labels
y_prob = model.predict_proba(X_test_sc)[:, 1]  # probability of Yes
```

### Step 11 — Evaluate (`evaluator.py`)

All plots and metrics computed and saved to the `outputs/` folder.

### Step 12 — Save Outputs (`model.py` + `evaluator.py`)

Model coefficients, CV scores, and all metrics saved to JSON files.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions / total |
| ROC-AUC | Area under ROC curve — measures discriminative ability |
| Average Precision | Area under Precision-Recall curve — better for imbalanced data |
| F1-Score | Harmonic mean of Precision and Recall for the positive class |
| Sensitivity | Recall for Yes — how many actual subscribers were found |
| Specificity | Recall for No — how many actual non-subscribers were correctly rejected |
| Confusion Matrix | Full breakdown of TP, TN, FP, FN |

> **Why ROC-AUC over Accuracy?** A classifier predicting "No" for everyone achieves ~88% accuracy but zero utility. ROC-AUC and Average Precision are far more meaningful metrics for imbalanced classification.

---

## Results

### Confusion Matrix

```
              Predicted
               No     Yes
Actual  No   [6773   1212]
Actual  Yes  [ 195    863]

True  Negatives (TN): 6,773  — Correctly predicted No
False Positives (FP): 1,212  — Predicted Yes, Actually No
False Negatives (FN):   195  — Predicted No, Actually Yes
True  Positives (TP):   863  — Correctly predicted Yes
```

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **84.44%** |
| **ROC-AUC** | **0.9076** |
| **Average Precision** | **0.5370** |
| **F1-Score (Yes)** | **0.5509** |
| **Sensitivity / Recall** | **81.6%** |
| **Specificity** | **84.8%** |
| **CV ROC-AUC (5-Fold)** | **0.9102 ± 0.0019** |

### Classification Report

```
              precision    recall  f1-score   support

    No  (0)       0.97      0.85      0.91      7985
    Yes (1)       0.42      0.82      0.55      1058

    accuracy                           0.84      9043
   macro avg       0.69      0.83      0.73      9043
weighted avg       0.91      0.84      0.86      9043
```

---

## Key Findings

1. **Call duration** is the strongest predictor — longer conversations strongly indicate customer interest and willingness to subscribe.
2. **Previous campaign success** (`poutcome_success`) is a powerful positive signal — customers who subscribed before are likely to subscribe again.
3. **Housing loan** (`housing`) is a negative predictor — customers with a housing loan are less likely to subscribe, likely due to existing financial commitments.
4. **Month effects** are significant — March, September, October and December outperform May for subscription rates.
5. **Random oversampling** substantially improved Recall for the positive class without sacrificing overall accuracy.
6. **Stable cross-validation** — std of only ±0.0019 across 5 folds confirms the model generalises well and is not overfitting.

---

## Output Files

| File | Description |
|------|-------------|
| `eda_overview.png` | 6-panel EDA visualisation |
| `confusion_matrix.png` | Predicted vs actual class breakdown |
| `roc_curve.png` | ROC curve with AUC and optimal threshold |
| `precision_recall.png` | Precision-Recall curve with AP score |
| `feature_importance.png` | Top 20 features by coefficient magnitude |
| `cv_scores.png` | 5-fold cross-validation ROC-AUC bar chart |
| `metrics.json` | All evaluation metrics in JSON format |
| `model_info.json` | Model coefficients and CV scores in JSON format |

---

## Setup & Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset

Place `bank-full.csv` inside the `data/` folder.

### Run

```bash
cd src
python main.py
```

The pipeline will execute all 12 steps sequentially, printing detailed output at each stage and saving all plots and JSON files to `outputs/`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas / numpy | Data loading and manipulation |
| scikit-learn | ML pipeline: encoding, split, oversample, scale, model, metrics |
| matplotlib / seaborn | All visualisations |

---

## Author

**Minar Im**
GitHub: [@MinarIm](https://github.com/MinarIm)
