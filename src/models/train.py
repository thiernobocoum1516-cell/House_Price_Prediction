# TRAIN.PY (VERSION PRODUCTION)

import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X_train = pd.read_csv("../data/processed/X_train_full.csv")
X_test = pd.read_csv("../data/processed/X_test_full.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

# Train ONLY best model
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    verbose=False
)

model.fit(X_train, y_train)

# Evaluation rapide
pred = model.predict(X_test)

print("R2:", r2_score(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred, squared=False))

# Save model
joblib.dump(model, "../models/best_model.pkl")