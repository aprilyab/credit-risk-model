import pickle
from joblib import load
import os
from fastapi import FastAPI
from app.pydantic_models import PredictRequest, PredictResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")

# Load model once at startu
model = load(MODEL_PATH)

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def read_root():
    return {"status": "Credit Risk API is running!"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert request dict into proper model input
    X = [list(request.features.values())]  # [[feat1, feat2, ...]]
    proba = model.predict_proba(X)[0, 1]   # probability of default
    return PredictResponse(risk_probability=float(proba))