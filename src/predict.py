"""
Prediction module for credit risk model with preprocessing.

Provides:
1. Model loading (lazy singleton pattern)
2. Full pipeline prediction (preprocessing + model)
3. Single-customer prediction with PD, score, and loan terms
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

# ------------------------------
# Scorecard Class
# ------------------------------
class Scorecard:
    """Convert predicted probabilities into a credit score using PDO."""
    def __init__(self, base_score: float = 600.0, p_at_base: float = 0.5, pdo: float = 50.0):
        self.base_score = base_score
        self.odds0 = p_at_base / (1 - p_at_base)
        self.factor = pdo / np.log(2)

    def prob_to_score(self, p: float) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        odds = (1 - p) / p
        return float(self.base_score + self.factor * np.log(odds / self.odds0))

# ------------------------------
# Loan policy helper
# ------------------------------
def suggest_terms(pd: float, avg_amount: float) -> Dict[str, float]:
    """Suggest loan terms based on PD and past average transaction amount."""
    max_limit = max(100, avg_amount * (1 - pd))
    max_days = int(30 * (1 - pd))
    return {"suggested_limit": round(max_limit, 2),
            "suggested_duration_days": max_days}

# ------------------------------
# Globals for singleton
# ------------------------------
_model_pipeline = None

# ------------------------------
# Load full pipeline (preprocessor + model)
# ------------------------------
def load_model(path: str = "models/best_model.pkl"):
    global _model_pipeline
    if _model_pipeline is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        _model_pipeline = joblib.load(path)
        print(f"✅ Pipeline loaded from {path}")
    return _model_pipeline

# ------------------------------
# Predict for single customer
# ------------------------------
def predict_one(payload: Dict[str, Any]) -> Dict[str, float]:
    pipeline = load_model()  # full pipeline

    # Convert payload to DataFrame
    X = pd.DataFrame([payload])

    # Fill missing columns with 0 to match training features
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        for col in preprocessor.feature_names_in_:
            if col not in X.columns:
                X[col] = 0

    # Predict probability of default
    proba = float(pipeline.predict_proba(X)[:, 1][0])

    # Convert to score
    scorecard = Scorecard()
    score = scorecard.prob_to_score(proba)

    # Suggest loan terms
    terms = suggest_terms(proba, payload.get("avg_amount", 100.0))

    return {"pd": round(proba, 6), "score": round(score, 1), **terms}

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    test_payload = {
        "Recency": 5,
        "Frequency": 10,
        "Monetary": 500,
        "total_amount": 500,
        "avg_amount": 50,
        "std_amount": 10,
        "txn_count": 10,
        "provider_count": 2,
        "product_count": 3,
        "category_count": 2,
        "channel_count": 1,
        "fraud_rate": 0,
        "hour_mean": 14,
        "hour_std": 2,
        "day_mean": 15,
        "day_nunique": 1,
        "month_mean": 8,
        "month_nunique": 1,
        "year_mean": 2025,
        "year_nunique": 1,
    }

    result = predict_one(test_payload)
    print("Prediction result:\n", result)
