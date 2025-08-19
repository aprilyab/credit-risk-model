"""
Train and Evaluate Credit Risk Model with SHAP Explainability
=============================================================

Features:
---------
1. Loads processed data from src/data_processing.py
2. Handles missing values with imputation
3. Balances dataset using SMOTE
4. Hyperparameter tuning with GridSearchCV
5. MLflow experiment tracking & model registry integration
6. Supports multiple models (LogisticRegression, RandomForest)
7. Generates SHAP explainability plots automatically
8. Saves best model locally
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn
from data_processing import build_features_pipeline, ID_COL, TARGET_COL
import shap
import matplotlib.pyplot as plt
import joblib

# ========================
# CONSTANTS
# ========================
DATA_PATH = "data/processed/processed_data.csv"
EXPERIMENT_NAME = "Credit_Risk_Model"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SHAP_DIR = "reports/shap"
MODEL_DIR = "models"

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================
# LOAD DATA
# ========================
def load_features(path: str):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset!")
    X = df.drop([ID_COL, TARGET_COL], axis=1)
    y = df[TARGET_COL]
    return X, y

# ========================
# TRAIN & EVALUATE
# ========================
def train_and_evaluate(X, y):
    # Split data (stratify to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = build_features_pipeline(numeric=numeric_features, categorical=categorical_features)

    # Define models and hyperparameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None]
            }
        }
    }

    best_models = {}
    mlflow.set_experiment(EXPERIMENT_NAME)

    for name, cfg in models.items():
        print(f"🔹 Training {name}...")

        # Pipeline with preprocessing + SMOTE + model
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", cfg["model"])
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["params"],
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            error_score="raise"
        )

        with mlflow.start_run(run_name=name):
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            y_pred_proba = grid.predict_proba(X_test)[:, 1]

            # Metrics
            from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print(f"{name} ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f}")
            print(report)

            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(
                grid.best_estimator_, name="model", registered_model_name=name,
                input_example=X_train.iloc[:5]
            )

            best_models[name] = grid.best_estimator_



    # ========================
    # Save the best model locally
    # ========================
    best_model_name = max(best_models, key=lambda k: roc_auc_score(y_test, best_models[k].predict_proba(X_test)[:,1]))
    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_models[best_model_name], best_model_path)
    print(f"📦 Best model '{best_model_name}' saved locally at {best_model_path}")

    return best_models

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("🚀 Starting training pipeline...")
    X, y = load_features(DATA_PATH)

    # Check class balance
    print("Class distribution:\n", y.value_counts())

    if len(y.value_counts()) < 2:
        raise ValueError("Target has less than 2 classes. Cannot train model.")

    best_models = train_and_evaluate(X, y)
    print("✅ Training completed. Best models saved to MLflow registry.")
   
