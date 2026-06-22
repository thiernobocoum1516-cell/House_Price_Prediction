import numpy as np
import joblib

from src.data.loader import load_train_data
from src.features.builder import create_all_features
from api.predictor import predict_house_price


# ============================================
# 1. TEST LOADER
# ============================================

def test_loader():

    df = load_train_data()

    assert df is not None
    assert df.shape[0] > 0
    assert "SalePrice" in df.columns


# ============================================
# 2. TEST FEATURE ENGINEERING
# ============================================

def test_feature_engineering():

    df = load_train_data()
    df_feat = create_all_features(df)

    assert df_feat.shape[0] > 0

    assert df_feat.isnull().sum().sum() < df_feat.size * 0.5

    assert df_feat.select_dtypes(include=["object"]).shape[1] >= 0


# ============================================
# 3. TEST FEATURES.PKL EXISTE
# ============================================

def test_features_file_exists():

    features = joblib.load("models/features.pkl")

    assert isinstance(features, list)
    assert len(features) > 0


# ============================================
# 4. TEST PREDICTION PIPELINE
# ============================================

def test_prediction_pipeline():

    sample = {
    "OverallQual": 7,
    "OverallCond": 5,
    "ExterQual": "TA",
    "GrLivArea": 1500,
    "LotArea": 8000,
    "TotalBsmtSF": 800,
    "LotFrontage": 70,
    "YearBuilt": 2005,
    "YearRemodAdd": 2010,
    "GarageCars": 2,
    "GarageArea": 400,
    "FullBath": 2,
    "HalfBath": 1,
    "Fireplaces": 1,
    "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr",
    "HouseStyle": "1Story",
    "BldgType": "1Fam",
    "MSZoning": "RL",
    "SaleCondition": "Normal"
}
    pred = predict_house_price(sample)

    assert isinstance(pred, float)
    assert pred > 0


# ============================================
# 5. TEST INF VALUES
# ============================================

def test_no_infinite_values():

    df_feat = create_all_features(load_train_data())

    assert np.isinf(
        df_feat.select_dtypes(include=np.number)
    ).sum().sum() == 0