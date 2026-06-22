import numpy as np

from src.data.loader import load_train_data
from src.features.builder import create_all_features


def test_loader():

    df = load_train_data()

    assert df is not None
    assert df.shape[0] > 0
    assert "SalePrice" in df.columns


def test_feature_engineering():

    df = load_train_data()
    df_feat = create_all_features(df)

    expected_features = [
        "TotalSF",
        "HouseAge",
        "RemodAge"
    ]

    for feature in expected_features:
        assert feature in df_feat.columns

    assert df_feat.shape[0] > 0

    assert (
        df_feat.isnull().sum().sum()
        < df_feat.size * 0.5
    )

    assert df_feat["TotalSF"].dtype != "object"


def test_no_infinite_values():

    df_feat = create_all_features(
        load_train_data()
    )

    assert (
        np.isinf(
            df_feat.select_dtypes(include=np.number)
        ).sum().sum()
        == 0
    )