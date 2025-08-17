import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from predict import predict_one, suggest_terms, Scorecard

class TestPredictHelpers(unittest.TestCase):
    def setUp(self):
        self.test_payload = {
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

    # ------------------------------
    # predict_one tests
    # ------------------------------
    def test_predict_one_returns_keys(self):
        """Check predict_one returns the correct keys."""
        result = predict_one(self.test_payload)
        expected_keys = {"pd", "score", "suggested_limit", "suggested_duration_days"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_predict_one_values_are_reasonable(self):
        """Check predicted values are within expected ranges."""
        result = predict_one(self.test_payload)
        self.assertGreaterEqual(result["pd"], 0)
        self.assertLessEqual(result["pd"], 1)
        self.assertGreater(result["score"], 0)
        self.assertGreaterEqual(result["suggested_limit"], 0)
        self.assertGreaterEqual(result["suggested_duration_days"], 0)

    # ------------------------------
    # suggest_terms tests
    # ------------------------------
    def test_suggest_terms_edge_pd_zero(self):
        terms = suggest_terms(0.0, 100.0)
        self.assertEqual(terms["suggested_limit"], 100.0)
        self.assertEqual(terms["suggested_duration_days"], 30)

    def test_suggest_terms_edge_pd_one(self):
        terms = suggest_terms(1.0, 100.0)
        self.assertEqual(terms["suggested_limit"], 100.0)  # minimum limit
        self.assertEqual(terms["suggested_duration_days"], 0)

    # ------------------------------
    # Scorecard tests
    # ------------------------------
    def test_scorecard_prob_to_score_bounds(self):
        sc = Scorecard()
        self.assertIsInstance(sc.prob_to_score(0.5), float)
        self.assertIsInstance(sc.prob_to_score(0.0), float)
        self.assertIsInstance(sc.prob_to_score(1.0), float)

    def test_scorecard_monotonicity(self):
        """Higher PD should result in lower score."""
        sc = Scorecard()
        low_pd_score = sc.prob_to_score(0.1)
        high_pd_score = sc.prob_to_score(0.9)
        self.assertGreater(low_pd_score, high_pd_score)

if __name__ == "__main__":
    unittest.main()
