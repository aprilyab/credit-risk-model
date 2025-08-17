import pickle
import os
from fastapi import FastAPI
from app.pydantic_models import PredictRequest, PredictResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")

# Load model once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

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
