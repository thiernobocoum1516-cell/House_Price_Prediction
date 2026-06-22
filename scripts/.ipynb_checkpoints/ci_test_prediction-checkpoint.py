from api.predictor import predict_house_price

def main():

    sample = {
        'OverallQual': 7,
        'OverallCond': 5,
        'ExterQual': 'TA',
        'GrLivArea': 1500,
        'LotArea': 8000,
        'TotalBsmtSF': 800,
        'LotFrontage': 70,
        'YearBuilt': 2005,
        'YearRemodAdd': 2010,
        'GarageCars': 2,
        'GarageArea': 400,
        'FullBath': 2,
        'HalfBath': 1,
        'Fireplaces': 1,
        'BedroomAbvGr': 3,
        'Neighborhood': 'CollgCr',
        'HouseStyle': '1Story',
        'BldgType': '1Fam',
        'MSZoning': 'RL',
        'SaleCondition': 'Normal'
    }

    pred = predict_house_price(sample)

    print("\n==============================")
    print(" SAMPLE PREDICTION CHECK")
    print("==============================")
    print("Input OK")
    print(f"Prediction: {pred}")
    print("==============================\n")

    assert isinstance(pred, float), "Prediction must be float"

if __name__ == "__main__":
    main()