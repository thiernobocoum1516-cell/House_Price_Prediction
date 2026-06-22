from pathlib import Path
import pandas as pd
import joblib

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ============================================
# CHEMINS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================
# CHARGEMENT DES DONNÉES
# ============================================

def load_data():

    X_train = pd.read_csv(
        BASE_DIR / "data" / "processed" / "X_train_full.csv"
    )

    X_test = pd.read_csv(
        BASE_DIR / "data" / "processed" / "X_test_full.csv"
    )

    y_train = pd.read_csv(
        BASE_DIR / "data" / "processed" / "y_train.csv"
    ).values.ravel()

    y_test = pd.read_csv(
        BASE_DIR / "data" / "processed" / "y_test.csv"
    ).values.ravel()

    return X_train, X_test, y_train, y_test


# ============================================
# ENTRAINEMENT
# ============================================

def main():

    X_train, X_test, y_train, y_test = load_data()

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        verbose=False
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(f"R²   : {r2_score(y_test, pred):.4f}")
    print(f"RMSE : {mean_squared_error(y_test, pred, squared=False):.2f}")

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(
        model,
        model_dir / "best_model.pkl"
    )

    print("\nModèle sauvegardé : models/best_model.pkl")


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    main()