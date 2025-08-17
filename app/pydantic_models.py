from pydantic import BaseModel
from typing import Dict

class PredictRequest(BaseModel):
    features: Dict[str, float]  

class PredictResponse(BaseModel):
    risk_probability: float
