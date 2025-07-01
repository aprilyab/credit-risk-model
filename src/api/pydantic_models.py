from pydantic import BaseModel

class CustomerData(BaseModel):
    amount: float
    channel: str
    recency: int
    frequency: int
    monetary: float

class PredictionResponse(BaseModel):
    risk_score: float
    risk_class: str
