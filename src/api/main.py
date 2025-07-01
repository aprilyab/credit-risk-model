from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load your trained credit risk model
with open(r"C:\Users\user\Desktop\credit-risk-model\models\credit_risk_model.joblib", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Input schema
class CreditInput(BaseModel):
    amount: float
    channel: str
    recency: int
    frequency: int
    monetary: float

@app.post("/predict/")
async def predict(input_data: CreditInput):
    # Convert input into a DataFrame (must match training schema)
    data = pd.DataFrame([{
        "amount": input_data.amount,
        "channel": input_data.channel,
        "recency": input_data.recency,
        "frequency": input_data.frequency,
        "monetary": input_data.monetary
    }])

    # Encode or process 'channel' if your model expects it
    # For example, one-hot encode or label encode 'channel'
    # (You must use same preprocessing used during training)

    # Make prediction
    prediction = model.predict_proba(data)[0][1]  # probability of class 1
    risk_class = "high" if prediction > 0.5 else "low"

    return {
        "risk_score": round(float(prediction), 2),
        "risk_class": risk_class
    }

