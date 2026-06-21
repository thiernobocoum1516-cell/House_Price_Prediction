#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# IMPORTS PROPRES
from src.data.loader import load_train_data
from src.features.building import Preprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(test_size=0.2, random_state=42):

    df = load_train_data()

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_model(X_train, y_train, config):

    model = CatBoostRegressor(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"],
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    return model


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, X, y, name="set"):

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    logger.info(f"{name} -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


# =============================================================================
# PIPELINE TRAIN
# =============================================================================
def train(config_path="config.yaml"):

    # ---------------- LOAD CONFIG ----------------
    if Path(config_path).exists():
        config = yaml.safe_load(open(config_path))
    else:
        config = {
            "data": {
                "test_size": 0.2,
                "random_state": 42
            },
            "model": {
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3
            },
            "output": {
                "model_dir": "models"
            }
        }

    # ---------------- LOAD DATA ----------------
    X_train, X_val, y_train, y_val = load_data(
        config["data"]["test_size"],
        config["data"]["random_state"]
    )

    # ---------------- PREPROCESSING ----------------
    preprocessor = Preprocessor()
    preprocessor.fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)

    # ---------------- TRAIN MODEL ----------------
    model = train_model(X_train, y_train, config["model"])

    # ---------------- EVALUATION ----------------
    train_metrics = evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluate(model, X_val, y_val, "Validation")

    # ---------------- SAVE ARTIFACTS ----------------
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")
    joblib.dump(preprocessor, run_dir / "preprocessor.pkl")

    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "config": config
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Artifacts saved in {run_dir}")

    return model, preprocessor, metrics


# =============================================================================
# ENTRY
# =============================================================================
if __name__ == "__main__":
    train()