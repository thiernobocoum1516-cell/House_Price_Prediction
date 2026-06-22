from pydantic import BaseModel

class HouseInput(BaseModel):

    # === Quality / Condition ===
    OverallQual: int
    OverallCond: int
    ExterQual: str   # ⭐ NEW (le 20e)

    # === Surface ===
    GrLivArea: float
    LotArea: float
    TotalBsmtSF: float
    LotFrontage: float

    # === Dates ===
    YearBuilt: int
    YearRemodAdd: int

    # === Garage ===
    GarageCars: int
    GarageArea: float

    # === Rooms ===
    FullBath: int
    HalfBath: int
    Fireplaces: int
    BedroomAbvGr: int

    # === Categorical ===
    Neighborhood: str
    HouseStyle: str
    BldgType: str
    MSZoning: str
    SaleCondition: str