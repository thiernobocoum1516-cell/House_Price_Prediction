from pathlib import Path
import json
import joblib
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ============================================
# CHEMINS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================
# CHARGEMENT DES DONNÉES
# ============================================

def load_data():
    X_train = pd.read_csv(DATA_PATH / "X_train_full.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test_full.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


# ============================================
# CHARGEMENT DES PARAMÈTRES
# ============================================

def load_params():
    params_path = MODEL_DIR / "best_params.json"

    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
        print(" Hyperparamètres chargés depuis best_params.json")
    else:
        print(" best_params.json introuvable → fallback")
        params = {
            "iterations": 300,
            "learning_rate": 0.1,
            "depth": 6
        }

    return params


# ============================================
# TRAINING
# ============================================

def train_model(X_train, y_train, params):

    # garder uniquement les vrais hyperparams CatBoost
    allowed_keys = {
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "loss_function"
    }

    clean_params = {k: v for k, v in params.items() if k in allowed_keys}

    model = CatBoostRegressor(
        **clean_params,
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    return model

# ============================================
# EVALUATION
# ============================================

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    print("\n PERFORMANCE")
    print(f" R²   : {r2:.4f}")
    print(f" RMSE : {rmse:.2f}")

    return r2, rmse


# ============================================
# SAVE ARTIFACTS (MODEL + FEATURES + PARAMS)
# ============================================

def save_artifacts(model, X_train, params, r2, rmse):

    # 1. modèle
    joblib.dump(model, MODEL_DIR / "best_model.pkl")

    # 2. features (CRITIQUE POUR L'API)
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, MODEL_DIR / "features.pkl")

    # 3. paramètres utilisés
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(params, f, indent=4)

    # 4. metadata (optionnel mais propre)
    metadata = {
        "model_type": "CatBoostRegressor",
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse)
        },
        "n_features": len(feature_names)
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n Artifacts sauvegardés :")
    print(" - best_model.pkl")
    print(" - features.pkl")
    print(" - best_params.json")
    print(" - model_metadata.json")


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def main():

    print("\n" + "="*60)
    print(" TRAINING PIPELINE - CATBOOST CLEAN MLOPS")
    print("="*60)

    # 1. data
    X_train, X_test, y_train, y_test = load_data()

    # 2. params
    params = load_params()

    # 3. train
    model = train_model(X_train, y_train, params)

    # 4. eval
    r2, rmse = evaluate(model, X_test, y_test)

    # 5. save everything
    save_artifacts(model, X_train, params, r2, rmse)

    print("\n Pipeline terminé avec succès")


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    main()