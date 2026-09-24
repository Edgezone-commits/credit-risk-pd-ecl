from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_PATH = Path("data/raw/credit_default.csv")

FEATURES = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
             "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
            + [f"BILL_AMT{i}" for i in range(1, 7)]
            + [f"PAY_AMT{i}" for i in range(1, 7)])

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Download the UCI Default of Credit Card Clients dataset once,
    then load from local cache on subsequent calls."""
    if path.exists():
        return pd.read_csv(path)

    ds = fetch_ucirepo(id=350)
    X, y = ds.data.features.copy(), ds.data.targets.copy()

    if list(X.columns)[0] == "X1":        # UCI sometimes returns generic names
        X.columns = FEATURES
    y.columns = ["default"]

    df = pd.concat([X, y], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and recode undocumented category values."""
    df = df.drop_duplicates().reset_index(drop=True)

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df[pay_cols] = df[pay_cols].replace({-2: -1, 0: -1})

    # Recoding can create new duplicates (e.g. two rows that only
    # differed by an undocumented category code now match exactly).
    df = df.drop_duplicates().reset_index(drop=True)

    return df