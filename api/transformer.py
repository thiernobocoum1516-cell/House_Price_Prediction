import pandas as pd
from src.features.builder import create_all_features
import pandas as pd


EXPECTED_COLS = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
    'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
    'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
    'QualityArea', 'TotalSF', 'HouseAge', 'RemodAge', 'TotalBath',
    'TotalPorchSF'
]


def transform_input(data: dict):

    df = pd.DataFrame([data])

    # =========================
    # DEFAULT VALUES (robustes)
    # =========================
    defaults = {
        "MSSubClass": 20,
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
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "LowQualFinSF": 0,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 6,
        "Functional": "Typ",
        "FireplaceQu": "TA",
        "GarageType": "Attchd",
        "GarageYrBlt": 2000,
        "GarageFinish": "Unf",
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
        "SaleType": "WD"
    }

    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # =========================
    # FEATURES ENGINEERED
    # =========================
    df = create_all_features(df)

    # =========================
    # ALIGNEMENT EXACT MODELE
    # =========================
    df = df.reindex(columns=EXPECTED_COLS)

    return df