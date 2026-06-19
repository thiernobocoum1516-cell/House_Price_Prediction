import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def load_train_data():
    return pd.read_csv(BASE_DIR / "data" / "train.csv")

def load_test_data():
    return pd.read_csv(BASE_DIR / "data" / "test.csv")