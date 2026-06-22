import pandas as pd
import joblib
from pathlib import Path
from src.features.builder import create_all_features


# ============================================
# PATH FEATURES (IMPORTANT)
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"


# ============================================
# LOAD FEATURES (SOURCE UNIQUE DE VÉRITÉ)
# ============================================

def load_features():
    return joblib.load(FEATURES_PATH)


# ============================================
# TRANSFORM INPUT
# ============================================

def transform_input(data: dict):

    df = pd.DataFrame([data])

    # ========================================
    # 1. COMPLETER VARIABLES MANQUANTES
    # ========================================

    # defaults minimal API (UNIQUEMENT INPUT SIDE)
    defaults = {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 70,
        "Street": "Pave",
        "Alley": "None",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 0,
        "TotalBsmtSF": 0,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 0,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 6,
        "Functional": "Typ",
        "FireplaceQu": "TA",
        "GarageType": "Attchd",
        "GarageYrBlt": 2000,
        "GarageFinish": "Unf",
        "GarageCars": 0,
        "GarageArea": 0,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 0,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2010,
        "SaleType": "WD",
        "SaleCondition": "Normal"
    }

    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # ========================================
    # 2. FEATURE ENGINEERING (IDENTIQUE TRAIN)
    # ========================================

    df = create_all_features(df)

    # ========================================
    # 3. ALIGNEMENT FINAL (CRITIQUE)
    # ========================================

    features = load_features()
    df = df.reindex(columns=features, fill_value=0)

    return df