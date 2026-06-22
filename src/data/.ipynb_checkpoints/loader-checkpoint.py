import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()


def load_train_data():
    df = pd.read_csv(BASE_DIR / "data" / "train.csv")
    return postprocess(df)


def load_test_data():
    df = pd.read_csv(BASE_DIR / "data" / "test.csv")
    return postprocess(df)


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # suppression Id (robuste)
    for col in ["Id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # NaN → "None" pour catégorielles
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("None")

    return df