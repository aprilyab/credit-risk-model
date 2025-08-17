# src/data_processing.py

"""
Data Processing Pipeline for Transaction Data
=============================================

This module provides robust, automated, and reproducible data processing utilities
to transform raw transactional data into model-ready datasets.

Main Features
-------------
1. Load and type-cast raw data from CSV files.
2. Compute RFM (Recency, Frequency, Monetary) features.
3. Generate user-level aggregate features.
4. Extract temporal features (hour, day, month, year).
5. Encode categorical variables using One-Hot Encoding.
6. Handle missing values with imputation strategies.
7. Normalize/standardize numeric features.
8. Label customers as "high risk" based on RFM clustering.
9. Save processed datasets to efficient formats (CSV, Parquet).

Typical Workflow
----------------
df_raw = load_raw("data/raw/data.csv")
rfm_df = RFMTransformer().fit_transform(df_raw)
agg_df = make_aggregate_features(df_raw)
features_df = join_features(rfm_df, agg_df)
features_df = add_temporal_features(df_raw, features_df)
features_df = features_pipeline.fit_transform(features_df)
save_processed(features_df, "data/processed/processed_data.csv")
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# ========================
# CONSTANTS
# ========================
DATETIME_COL = "TransactionStartTime"
ID_COL = "CustomerId"
TARGET_COL = "is_high_risk"


# ========================
# RAW DATA LOADING
# ========================
def load_raw(path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV and parse datetime column."""
    df = pd.read_csv(path)
    if DATETIME_COL in df.columns:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    return df


# ========================
# RFM TRANSFORMER
# ========================
class RFMTransformer(BaseEstimator, TransformerMixin):
    """Transformer to compute Recency, Frequency, Monetary (RFM) features."""

    def __init__(self, snapshot_date: Optional[pd.Timestamp] = None):
        self.snapshot_date = snapshot_date

    def fit(self, X: pd.DataFrame, y=None):
        if self.snapshot_date is None:
            self.snapshot_date = pd.Timestamp(X[DATETIME_COL].max()) + pd.Timedelta(days=1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Monetary"] = df.get("Value", df["Amount"]).abs()

        rfm = df.groupby(ID_COL).agg(
            Recency=(DATETIME_COL, lambda s: (self.snapshot_date - s.max()).days),
            Frequency=("TransactionId", "nunique"),
            Monetary=("Monetary", "sum"),
        ).reset_index()
        return rfm


# ========================
# AGGREGATE FEATURES
# ========================
def make_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate user-level aggregates: transaction stats, product variety, fraud rate, time stats."""
    df2 = df.copy()
    df2["abs_amount"] = df2.get("Value", df2["Amount"]).abs()

    aggs = df2.groupby(ID_COL).agg(
        total_amount=("abs_amount", "sum"),
        avg_amount=("abs_amount", "mean"),
        std_amount=("abs_amount", "std"),
        txn_count=("TransactionId", "nunique"),
        provider_count=("ProviderId", "nunique"),
        product_count=("ProductId", "nunique"),
        category_count=("ProductCategory", "nunique"),
        channel_count=("ChannelId", "nunique"),
        fraud_rate=("FraudResult", "mean"),
    ).reset_index()

    # Add transaction hour-based features
    if DATETIME_COL in df2.columns:
        df2["hour"] = df2[DATETIME_COL].dt.hour
        hour_stats = df2.groupby(ID_COL)["hour"].agg(["mean", "std"]).reset_index()
        hour_stats = hour_stats.rename(columns={"mean": "hour_mean", "std": "hour_std"})
        aggs = aggs.merge(hour_stats, on=ID_COL, how="left")

    return aggs


# ========================
# TEMPORAL FEATURES
# ========================
def add_temporal_features(df_raw: pd.DataFrame, df_feats: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features (hour, day, month, year) and join to feature set."""
    if DATETIME_COL not in df_raw.columns:
        return df_feats

    temporal = df_raw.copy()
    temporal["hour"] = temporal[DATETIME_COL].dt.hour
    temporal["day"] = temporal[DATETIME_COL].dt.day
    temporal["month"] = temporal[DATETIME_COL].dt.month
    temporal["year"] = temporal[DATETIME_COL].dt.year

    temp_aggs = temporal.groupby(ID_COL)[["hour", "day", "month", "year"]].agg(
        ["mean", "nunique"]
    )
    temp_aggs.columns = ["_".join(col) for col in temp_aggs.columns]
    temp_aggs = temp_aggs.reset_index()

    return df_feats.merge(temp_aggs, on=ID_COL, how="left")


# ========================
# JOIN FEATURES
# ========================
def join_features(rfm: pd.DataFrame, aggs: pd.DataFrame) -> pd.DataFrame:
    """Merge RFM with aggregate features."""
    return rfm.merge(aggs, on=ID_COL, how="left")


# ========================
# CLUSTERING FOR RISK LABELING
# ========================
def label_high_risk_by_rfm(rfm_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Cluster customers with K-Means and label high-risk group."""
    feats = ["Recency", "Frequency", "Monetary"]
    rfm = rfm_df[feats].fillna(0.0).copy()
    Z = StandardScaler().fit_transform(rfm)

    km = KMeans(n_clusters=3, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(Z)

    rfm_df = rfm_df.copy()
    rfm_df["cluster"] = clusters
    clust_stats = rfm_df.groupby("cluster")[feats].mean()

    clust_stats = clust_stats.assign(
        score=clust_stats["Recency"].rank(ascending=False) +
              clust_stats["Frequency"].rank(ascending=True) +
              clust_stats["Monetary"].rank(ascending=True)
    )

    high_risk_cluster = clust_stats["score"].idxmax()
    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    return rfm_df[[ID_COL, "is_high_risk", "cluster"]]


# ========================
# PIPELINE FOR FEATURES
# ========================
def build_features_pipeline(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric & categorical features."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])


# ========================
# SAVE PROCESSED
# ========================
def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed dataset to CSV or Parquet depending on extension."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


# ========================
# MAIN SCRIPT
# ========================
if __name__ == "__main__":
    RAW_PATH = "data/raw/data.csv"
    PROCESSED_PATH = "data/processed/processed_data.csv"

    df_raw = load_raw(RAW_PATH)
    rfm_df = RFMTransformer().fit_transform(df_raw)
    agg_df = make_aggregate_features(df_raw)
    features_df = join_features(rfm_df, agg_df)
    features_df = add_temporal_features(df_raw, features_df)
    features_df = features_df.merge(label_high_risk_by_rfm(rfm_df), on=ID_COL, how="left")

    save_processed(features_df, PROCESSED_PATH)
    print(f"✅ Processed data saved at {PROCESSED_PATH}")
