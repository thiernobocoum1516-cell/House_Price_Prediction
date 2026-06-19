import pandas as pd
from src.data.loader import load_train_data
from src.features.builder import create_all_features


# =========================
# 1. TEST LOADER
# =========================
def test_loader():
    df = load_train_data()

    assert df is not None, "Le DataFrame est None"
    assert len(df) > 0, "Le dataset est vide"
    assert "SalePrice" in df.columns, "Colonne SalePrice manquante"


# =========================
# 2. TEST FEATURE ENGINEERING
# =========================
def test_feature_engineering():
    df = load_train_data()
    df_feat = create_all_features(df)

    # Vérifie que les nouvelles features existent
    assert "TotalSF" in df_feat.columns, "TotalSF manquant"
    assert "HouseAge" in df_feat.columns, "HouseAge manquant"
    assert "RemodAge" in df_feat.columns, "RemodAge manquant"

    # Vérifie que les données ne sont pas cassées
    assert df_feat.shape[0] > 0, "Feature engineering a supprimé toutes les lignes"


# =========================
# 3. SMOKE TEST GLOBAL
# =========================
def test_smoke():
    assert True