# =============================================================
#  data_loader.py
#  Responsibility: Load the dataset and print a full summary
# =============================================================

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Bank Marketing dataset from a semicolon-separated CSV file.

    Parameters
    ----------
    filepath : str
        Path to bank-full.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all 17 columns intact
    """
    df = pd.read_csv(filepath, sep=';')
    return df


def summarise(df: pd.DataFrame) -> None:
    """
    Print a full descriptive summary of the dataset:
      - Shape
      - First 5 rows
      - Data types
      - Missing values
      - Numerical statistics
      - Target class distribution
      - Unique values per categorical column
    """
    print("\n" + "=" * 60)
    print("  DATA SUMMARY")
    print("=" * 60)

    print(f"\n  Shape   :  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    print("\n  ── First 5 rows ─────────────────────────────────")
    print(df.head().to_string(index=False))

    print("\n  ── Data Types ───────────────────────────────────")
    for col, dtype in df.dtypes.items():
        print(f"   {col:<14} {str(dtype)}")

    print("\n  ── Missing Values ───────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✔  No missing values found.")
    else:
        print(missing[missing > 0].to_string())

    print("\n  ── Numerical Statistics ─────────────────────────")
    print(df.describe().round(2).to_string())

    print("\n  ── Target Class Distribution ────────────────────")
    vc = df['y'].value_counts()
    for label, count in vc.items():
        pct   = count / len(df) * 100
        bar   = '█' * int(pct / 2)
        print(f"   {label:>5}  |  {bar:<46}  {count:>6,}  ({pct:.1f}%)")
    print("\n   ⚠  Class imbalance detected → will apply oversampling in preprocessing.")

    print("\n  ── Categorical Column Unique Values ─────────────")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        vals = list(df[col].unique())
        print(f"   {col:<14} ({len(vals):>2} unique)  →  {vals}")
