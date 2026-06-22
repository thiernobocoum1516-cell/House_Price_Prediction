import joblib
from pathlib import Path
from api.transformer import transform_input


# ============================================
# PATH MODEL
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(MODEL_PATH)


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_house_price(data: dict):

    # 1. transformer input → 85 features
    df = transform_input(data)

    # 2. prediction
    pred = model.predict(df)

    return float(pred[0])