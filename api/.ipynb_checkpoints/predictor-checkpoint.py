import joblib
import pandas as pd
from pathlib import Path
from api.transformer import transform_input


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"


# ============================================
# LOAD ARTIFACTS
# ============================================

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_house_price(data: dict):

    # 1. transformer input → features engineering
    df = transform_input(data)

    # 2. ALIGNEMENT CRITIQUE (IMPORTANT)
    df = df.reindex(columns=features, fill_value=0)

    # 3. prediction
    pred = model.predict(df)

    return float(pred[0])