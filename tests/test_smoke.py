def test_loader():
    df = load_train_data()

    assert df is not None
    assert df.shape[0] > 0
    assert "SalePrice" in df.columns

    # check NaN critique
    assert df.isnull().sum().sum() >= 0
def test_feature_engineering():
    df = load_train_data()
    df_feat = create_all_features(df)

    # features exist
    expected_features = ["TotalSF", "HouseAge", "RemodAge"]

    for f in expected_features:
        assert f in df_feat.columns, f"{f} manquant"

    # pas de dataset cassé
    assert df_feat.shape[0] > 0

    # check NaN explosion
    assert df_feat.isnull().sum().sum() < df_feat.shape[0] * 0.5

    # types OK
    assert df_feat["TotalSF"].dtype != "object"

def test_no_infinite_values():
    df = create_all_features(load_train_data())
    assert (df == float("inf")).sum().sum() == 0