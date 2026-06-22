from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.schema import HouseInput
from api.predictor import predict_house_price


app = FastAPI(
    title="🏠 House Price Prediction API",
    description="ML API powered by CatBoost + clean MLOps pipeline",
    version="1.0.0"
)


# ============================================
# HOME PAGE
# ============================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>House Price API</title>
        </head>
        <body style="font-family:Arial; text-align:center; margin-top:50px;">
            <h1>🏠 House Price Prediction API</h1>
            <p>CatBoost ML Pipeline - Clean Architecture</p>
            <p><a href="/docs">Swagger Docs</a></p>
        </body>
    </html>
    """


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict(data: HouseInput):

    prediction = predict_house_price(data.dict())

    return {
        "predicted_price_usd": round(prediction, 2)
    }