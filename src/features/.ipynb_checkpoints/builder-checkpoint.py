import numpy as np
import pandas as pd


def create_all_features(df):
    df_feat = df.copy()

    # =========================
    # 1. FEATURE TEMPORELLES
    # =========================
    df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]
    df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]
    df_feat["GarageAge"] = df_feat["YrSold"] - df_feat["GarageYrBlt"]

    # =========================
    # 2. SURFACE TOTALE
    # =========================
    df_feat["TotalSF"] = (
        df_feat["TotalBsmtSF"]
        + df_feat["1stFlrSF"]
        + df_feat["2ndFlrSF"]
    )

    df_feat["TotalPorchSF"] = (
        df_feat["OpenPorchSF"]
        + df_feat["EnclosedPorch"]
        + df_feat["3SsnPorch"]
        + df_feat["ScreenPorch"]
    )

    # =========================
    # 3. VARIABLES UTILITAIRES
    # =========================
    df_feat["HasGarage"] = (df_feat["GarageArea"] > 0).astype(int)
    df_feat["HasPool"] = (df_feat["PoolArea"] > 0).astype(int)
    df_feat["HasFireplace"] = (df_feat["Fireplaces"] > 0).astype(int)

    # =========================
    # 4. TRANSFORMATIONS LOG
    # =========================
    df_feat["LogLotArea"] = np.log1p(df_feat["LotArea"])
    df_feat["LogSalePrice"] = np.log1p(df_feat["SalePrice"])

    return df_feat