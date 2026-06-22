from src.data.loader import load_train_data
from src.features.builder import create_all_features
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

df = load_train_data()
df_feat = create_all_features(df)

X = df_feat.drop("SalePrice", axis=1)
y = df_feat["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("data/processed/X_train_full.csv", index=False)
X_test.to_csv("data/processed/X_test_full.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("DATASET BUILT OK")