from pathlib import Path
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.schema import HouseInput
from api.predictor import predict_house_price


# ============================================
# APP
# ============================================

app = FastAPI(
    title="🏠 House Price Prediction API",
    description="ML API powered by CatBoost + clean MLOps pipeline",
    version="1.0.0"
)

templates = Jinja2Templates(directory="api/templates")


# ============================================
# LOAD MODEL METADATA
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]

with open(BASE_DIR / "models" / "model_metadata.json", "r") as f:
    metadata = json.load(f)


# ============================================
# HOME PAGE
# ============================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model": metadata["model"],
            "r2": metadata["r2"],
            "rmse": metadata["rmse"],
            "n_features": metadata["n_features"]
        }
    )


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