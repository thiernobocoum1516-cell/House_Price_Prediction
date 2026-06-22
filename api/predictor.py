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
CAT_FEATURES_PATH = BASE_DIR / "models" / "cat_features.pkl"


# ============================================
# LOAD ARTIFACTS
# ============================================

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
cat_features = joblib.load(CAT_FEATURES_PATH)


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_house_price(data: dict):

    # 1. transformation + feature engineering
    df = transform_input(data)

    # 2. alignement colonnes (CRITIQUE)
    df = df.reindex(columns=features, fill_value=0)

    # 3. sécurité CatBoost : forcer les catégories en string
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 4. prediction
    pred = model.predict(df)

    return float(pred[0])