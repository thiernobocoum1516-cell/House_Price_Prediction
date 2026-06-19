import pandas as pd
from src.data.loader import load_data
from src.features.builder import create_all_features


# =========================
# 1. TEST LOADER
# =========================
def test_loader():
    df = load_data("data/train.csv")

    assert df is not None, "Le DataFrame est None"
    assert len(df) > 0, "Le dataset est vide"
    assert "SalePrice" in df.columns, "Colonne cible manquante"


# =========================
# 2. TEST FEATURES
# =========================
def test_feature_engineering():
    df = pd.read_csv("data/train.csv")
    df_feat = create_all_features(df)

    # Vérifie que des nouvelles features existent
    assert "TotalSF" in df_feat.columns, "Feature TotalSF manquante"
    assert "HouseAge" in df_feat.columns, "Feature HouseAge manquante"
    assert "RemodAge" in df_feat.columns, "Feature RemodAge manquante"

    # Vérifie que les données ne sont pas cassées
    assert df_feat.shape[0] > 0, "Feature engineering a supprimé toutes les lignes"


# =========================
# 3. TEST SMOKE GLOBAL
# =========================
def test_smoke():
    assert 1 == 1