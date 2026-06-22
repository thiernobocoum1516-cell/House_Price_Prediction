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
# CHARGEMENT DES HYPERPARAMÈTRES
# ============================================

def load_params():
    params_path = MODEL_DIR / "best_params.json"

    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
        print(" Hyperparamètres chargés depuis best_params.json")
    else:
        # fallback sécurisé
        print(" best_params.json introuvable → paramètres par défaut")
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

    model = CatBoostRegressor(
        **params,
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
# SAVE MODEL
# ============================================

def save_model(model):

    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(model, model_path)

    print(f"\n Modèle sauvegardé : {model_path}")


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def main():

    print("\n" + "="*60)
    print(" TRAINING PIPELINE - CATBOOST (MLOPS CLEAN)")
    print("="*60)

    # 1. Data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Params
    params = load_params()

    # 3. Train
    model = train_model(X_train, y_train, params)

    # 4. Eval
    evaluate(model, X_test, y_test)

    # 5. Save
    save_model(model)

    print("\n Pipeline terminé avec succès")


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    main()