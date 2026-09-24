import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive repayment-behavior and utilization features from the raw columns."""
    df = df.copy()

    # Repayment delay summary
    df["avg_delay"] = df[PAY_COLS].mean(axis=1)
    df["max_delay"] = df[PAY_COLS].max(axis=1)
    df["delay_trend"] = df["PAY_0"] - df["PAY_6"]  # positive = worsening

    # Credit utilization: bill amount relative to credit limit
    util_cols = []
    for i, bill_col in enumerate(BILL_COLS, start=1):
        col_name = f"util_ratio_{i}"
        df[col_name] = df[bill_col] / df["LIMIT_BAL"].replace(0, np.nan)
        util_cols.append(col_name)
    df["avg_util_ratio"] = df[util_cols].mean(axis=1)

    # Payment-to-bill ratio: how much of the bill did they actually pay
    ratio_cols = []
    for i, (pay_col, bill_col) in enumerate(zip(PAYAMT_COLS, BILL_COLS), start=1):
        col_name = f"pay_to_bill_ratio_{i}"
        # avoid divide-by-zero / extreme ratios on tiny or negative bills
        safe_bill = df[bill_col].where(df[bill_col] > 0, np.nan)
        df[col_name] = (df[pay_col] / safe_bill).clip(upper=5)
        ratio_cols.append(col_name)
    df["avg_pay_ratio"] = df[ratio_cols].mean(axis=1)

    # Fill NaNs created by zero/negative bills or limits with 0
    # (interpreted as "no utilization" / "no payment ratio to speak of")
    new_cols = util_cols + ratio_cols + ["avg_util_ratio", "avg_pay_ratio"]
    df[new_cols] = df[new_cols].fillna(0)

    return df