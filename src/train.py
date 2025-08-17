import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# ------------------------------
# Scorecard
# ------------------------------
class Scorecard:
    def __init__(self, base_score: float = 600.0, p_at_base: float = 0.5, pdo: float = 50.0):
        self.base_score = base_score
        self.odds0 = p_at_base / (1 - p_at_base)
        self.factor = pdo / np.log(2)

    def prob_to_score(self, p: float) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        odds = (1 - p) / p
        return float(self.base_score + self.factor * np.log(odds / self.odds0))

# ------------------------------
# Loan terms helper
# ------------------------------
def suggest_terms(pd: float, avg_amount: float) -> Dict[str, float]:
    max_limit = max(100, avg_amount * (1 - pd))
    max_days = int(30 * (1 - pd))
    return {"suggested_limit": round(max_limit, 2),
            "suggested_duration_days": max_days}

# ------------------------------
# Globals
# ------------------------------
_pipeline = None  # full pipeline with preprocessor + model
_required_features = None

# ------------------------------
# Load full pipeline (preprocessor + model)
# ------------------------------
def load_pipeline(path: str = "models/best_model.pkl"):
    global _pipeline, _required_features
    if _pipeline is None:
        _pipeline = joblib.load(path)
        _required_features = _pipeline.named_steps["preprocessor"].feature_names_in_
        print(f"✅ Pipeline loaded from {path}")
    return _pipeline

# ------------------------------
# Predict single customer
# ------------------------------
def predict_one(payload: Dict[str, Any]) -> Dict[str, float]:
    pipeline = load_pipeline()
    feature_names = _required_features

    # Wrap payload into DataFrame
    X = pd.DataFrame([payload])

    # Fill missing features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Reorder columns
    X = X[feature_names]

    # Predict probability of default directly using pipeline
    proba = float(pipeline.predict_proba(X)[:, 1][0])

    # Score and loan terms
    score = Scorecard().prob_to_score(proba)
    avg_amount = payload.get("avg_amount", 100.0)
    terms = suggest_terms(proba, avg_amount)

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
