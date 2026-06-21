#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

# 👉 IMPORT DU LOADER (IMPORTANT)
from src.data.loader import load_train_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    df_feat['QualityArea'] = df_feat['OverallQual'] * df_feat['GrLivArea']
    df_feat['TotalSF'] = df_feat['TotalBsmtSF'] + df_feat['1stFlrSF'] + df_feat['2ndFlrSF']
    df_feat['HouseAge'] = df_feat['YrSold'] - df_feat['YearBuilt']
    df_feat['RemodAge'] = df_feat['YrSold'] - df_feat['YearRemodAdd']
    df_feat['TotalBath'] = (
        df_feat['FullBath'] + 0.5 * df_feat['HalfBath'] +
        df_feat['BsmtFullBath'] + 0.5 * df_feat['BsmtHalfBath']
    )
    df_feat['TotalPorchSF'] = (
        df_feat['OpenPorchSF'] + df_feat['EnclosedPorch'] +
        df_feat['3SsnPorch'] + df_feat['ScreenPorch']
    )

    return df_feat


# =============================================================================
# PREPROCESSOR
# =============================================================================
class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputation_params = {}
        self.selected_features = None
        self.num_cols = None
        self.cat_cols = None
        self.fitted = False

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, df: pd.DataFrame, selected_features_path: str = None):

        logger.info("Fit preprocessing...")

        X = create_features(df)

        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        for col in ['Id', 'SalePrice']:
            if col in self.num_cols:
                self.num_cols.remove(col)
            if col in self.cat_cols:
                self.cat_cols.remove(col)

        self._impute(X)
        self._encode(X)

        self.scaler.fit(X[self.num_cols])

        # features selection
        if selected_features_path:
            with open(selected_features_path, "r") as f:
                self.selected_features = [line.strip() for line in f]
        else:
            self.selected_features = [c for c in X.columns if c not in ['Id', 'SalePrice']]

        self.fitted = True
        logger.info("Fit terminé")
        return self

    # =========================================================
    # TRANSFORM
    # =========================================================
    def transform(self, df: pd.DataFrame):

        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")

        logger.info("Transform...")

        X = create_features(df)

        self._apply_imputation(X)
        self._apply_encoding(X)

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        X_final = X[self.selected_features].copy()

        logger.info(f"Shape final: {X_final.shape}")
        return X_final

    # =========================================================
    # IMPUTATION FIT
    # =========================================================
    def _impute(self, X):
        self.imputation_params['median'] = X.median(numeric_only=True).to_dict()
        X.fillna(self.imputation_params['median'], inplace=True)

    # =========================================================
    # IMPUTATION TRANSFORM
    # =========================================================
    def _apply_imputation(self, X):
        X.fillna(self.imputation_params['median'], inplace=True)

    # =========================================================
    # ENCODING FIT
    # =========================================================
    def _encode(self, X):
        for col in self.cat_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.label_encoders[col] = le
            X[col] = le.transform(X[col])

    # =========================================================
    # ENCODING TRANSFORM
    # =========================================================
    def _apply_encoding(self, X):
        for col in self.cat_cols:
            le = self.label_encoders[col]
            X[col] = X[col].astype(str)
            known = set(le.classes_)
            X[col] = X[col].apply(lambda x: x if x in known else le.classes_[0])
            X[col] = le.transform(X[col])

    # =========================================================
    # SAVE / LOAD
    # =========================================================
    def save(self, path):
        joblib.dump(self, path)
        logger.info(f"Saved -> {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)


# =============================================================================
# FACTORY CLEAN (IMPORTANT POUR TRAIN.PY)
# =============================================================================
def get_preprocessor():
    return Preprocessor()