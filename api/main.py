from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api.schema import HouseInput
from api.predictor import predict_house_price

app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Machine Learning API using CatBoost + Feature Engineering",
    version="1.0.0"
)


# ============================================
# LANDING PAGE
# ============================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>House Price API</title>
            <style>
                body {
                    font-family: Arial;
                    background: linear-gradient(120deg, #1e3c72, #2a5298);
                    color: white;
                    text-align: center;
                    padding-top: 80px;
                }

                .box {
                    background: rgba(0,0,0,0.4);
                    padding: 30px;
                    border-radius: 20px;
                    width: 60%;
                    margin: auto;
                }

                h1 {
                    font-size: 42px;
                }

                p {
                    font-size: 18px;
                }

                a {
                    color: #FFD700;
                    font-weight: bold;
                    text-decoration: none;
                }
            </style>
        </head>

        <body>
            <div class="box">
                <h1>🏠 House Price Prediction API</h1>

                <p>Machine Learning model powered by CatBoost</p>

                <p>
                    20 User Inputs
                    →
                    Feature Engineering
                    →
                    85 Features
                    →
                    House Price Prediction
                </p>

                <p>
                    🚀 <a href="/docs">Open Swagger Documentation</a>
                </p>
            </div>
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
# PREDICTION
# ============================================

@app.post("/predict")
def predict(data: HouseInput):

    try:

        prediction = predict_house_price(data.dict())

        return {
            "predicted_price_usd": round(prediction, 2)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }