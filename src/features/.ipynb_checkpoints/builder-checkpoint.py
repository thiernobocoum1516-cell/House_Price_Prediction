#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV

# ============================================================
# IMPORT LOADER (source de vérité data)
# ============================================================
from src.data.loader import load_train_data, load_test_data

# ============================================================
# CONSTANTES EDA
# ============================================================
COLS_NONE = [
    "PoolQC", "MiscFeature", "Fence", "FireplaceQu", "Alley",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"]
    )

    return df


# ============================================================
# CLEANING INITIAL (aligné loader)
# ============================================================
def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sécurité Id (si présent)
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    # NaN → "None"
    df[COLS_NONE] = df[COLS_NONE].fillna("None")

    return df


# ============================================================
# FEATURE ENGINEER CLASS
# ============================================================
class FeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = None
        self.num_cols = None
        self.cat_cols = None
        self.feature_names = None

    # --------------------------------------------------------
    # FIT + TRANSFORM (TRAIN)
    # --------------------------------------------------------
    def fit_transform(self, X, y):

        # ===== features =====
        X = create_all_features(X)

        # ===== cleaning =====
        X = self._impute(X, fit=True)

        # ===== encoding =====
        X = self._encode(X, fit=True)

        # ===== numeric scaling =====
        self.num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.num_cols = [c for c in self.num_cols if c != "SalePrice"]

        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])

        # ===== feature selection =====
        self.feature_names = X.columns.tolist()

        print("LassoCV training...")
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X.values, y)

        self.selected_features = [
            self.feature_names[i]
            for i, coef in enumerate(lasso.coef_)
            if coef != 0
        ]

        print(f"{len(self.selected_features)} features sélectionnées")

        return X

    # --------------------------------------------------------
    # TRANSFORM (PREDICTION)
    # --------------------------------------------------------
    def transform(self, X):

        X = create_all_features(X)
        X = self._impute(X, fit=False)
        X = self._encode(X, fit=False)

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        # sécurité alignment
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        X = X[self.feature_names]

        return X

    # --------------------------------------------------------
    # IMPUTATION (version notebook)
    # --------------------------------------------------------
    def _impute(self, X, fit=False):
        X = X.copy()

        # LotFrontage par Neighborhood
        if "Neighborhood" in X.columns:
            if fit:
                self.neigh_medians = X.groupby("Neighborhood")["LotFrontage"].median()

            X["LotFrontage"] = X.apply(
                lambda r: self.neigh_medians.get(r["Neighborhood"], X["LotFrontage"].median())
                if pd.isna(r["LotFrontage"]) else r["LotFrontage"],
                axis=1
            )

        # MasVnrArea
        if "MasVnrType" in X.columns:
            X.loc[X["MasVnrType"].isna(), "MasVnrArea"] = 0

        X["MasVnrArea"] = X["MasVnrArea"].fillna(X["MasVnrArea"].median())

        # GarageYrBlt
        if "GarageType" in X.columns:
            X.loc[X["GarageType"].isna(), "GarageYrBlt"] = 0

        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])

        # Electrical
        mode = X["Electrical"].mode()[0]
        X["Electrical"] = X["Electrical"].fillna(mode).astype(str)

        # fallback
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna("None")

        return X

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------
    def _encode(self, X, fit=False):
        X = X.copy()

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.cat_cols = cat_cols

        for col in cat_cols:
            X[col] = X[col].astype(str)

            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        return X


# ============================================================
# PIPELINE ENTRY POINT
# ============================================================
def build(mode="train"):

    if mode == "train":
        df = load_train_data()
    else:
        df = load_test_data()

    df = initial_cleaning(df)
    df = create_all_features(df)

    return df


# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    df = build("train")
    print(df.shape)