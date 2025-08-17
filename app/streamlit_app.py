"""
Streamlit Dashboard for Credit Risk Proxy Project
-------------------------------------------------
File: app.py
Run: streamlit run app.py

Features:
- Upload raw transactions CSV (Xente-like) or use example/sample data
- Compute RFM (Recency, Frequency, Monetary) per customer
- Label high-risk customers using KMeans clustering
- Display cluster summary, RFM distributions, and a PCA scatter plot
- Optional: Run trained model predictions on aggregated users
- Optional: Show PD histogram and top feature importances (model-dependent)
- Optional: SHAP explainability for model predictions
- Single-customer prediction form with suggested loan terms

Notes:
- Expects a datetime column named 'TransactionStartTime'
- Expects a monetary column: either 'Value' or 'Amount'
- Place trained model at 'models/best_model.joblib' to enable predictions
"""

# ----------------- Imports -----------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------- Constants -----------------
MODEL_PATH = "models/best_model.joblib"

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk — RFM Proxy & Dashboard")
st.markdown(
    "Demo dashboard: compute RFM, cluster customers, label high-risk proxy, "
    "and optionally run model predictions and SHAP explainability."
)

# ----------------- Utility Functions -----------------
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV file with fallback encoding for special characters."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='latin1')


@st.cache_data
def compute_rfm(df: pd.DataFrame, id_col='CustomerId', date_col='TransactionStartTime',
                amount_cols=('Value','Amount')) -> pd.DataFrame:
    """
    Compute RFM metrics for each customer.
    Returns a DataFrame with Recency, Frequency, and Monetary.
    """
    d = df.copy()

    # Identify the monetary column
    amount_col = next((c for c in amount_cols if c in d.columns), None)
    if amount_col is None:
        raise ValueError(f"No amount column found among {amount_cols}")

    # Parse datetime column
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    if d[date_col].isna().all():
        raise ValueError(f"All values in {date_col} could not be parsed as datetime")

    snapshot = d[date_col].max() + pd.Timedelta(days=1)
    d['monetary_abs'] = d[amount_col].abs()

    # Aggregate RFM per customer
    grp = d.groupby(id_col).agg(
        Recency=(date_col, lambda s: (snapshot - s.max()).days),
        Frequency=('TransactionId', 'nunique') if 'TransactionId' in d.columns else (date_col, 'count'),
        Monetary=('monetary_abs', 'sum')
    ).reset_index()

    return grp


@st.cache_data
def label_high_risk_by_rfm(rfm_df: pd.DataFrame, random_state: int=42):
    """
    Cluster customers using KMeans and label the highest-risk cluster.
    Returns labeled RFM DataFrame and KMeans model.
    """
    feats = ['Recency','Frequency','Monetary']
    r = rfm_df.copy()
    r[feats] = r[feats].fillna(0.0)

    # Standardize features and cluster
    Z = StandardScaler().fit_transform(r[feats])
    km = KMeans(n_clusters=3, random_state=random_state, n_init='auto')
    clusters = km.fit_predict(Z)
    r['cluster'] = clusters

    # Determine high-risk cluster by ranking
    clust_stats = r.groupby('cluster')[feats].mean()
    score = (
        clust_stats['Recency'].rank(ascending=False) +
        clust_stats['Frequency'].rank(ascending=True) +
        clust_stats['Monetary'].rank(ascending=True)
    )
    high_risk_cluster = score.idxmax()
    r['is_high_risk'] = (r['cluster'] == high_risk_cluster).astype(int)

    return r, km


@st.cache_data
def aggregate_features_from_transactions(df: pd.DataFrame, id_col='CustomerId',
                                         date_col='TransactionStartTime') -> pd.DataFrame:
    """
    Aggregate transaction-level features to customer-level for model input.
    """
    d = df.copy()
    amount_col = 'Value' if 'Value' in d.columns else ('Amount' if 'Amount' in d.columns else None)
    if amount_col is None:
        raise ValueError("No Amount/Value column found.")

    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    d['abs_amount'] = d[amount_col].abs()

    # Aggregation
    agg = d.groupby(id_col).agg(
        total_amount=('abs_amount', 'sum'),
        avg_amount=('abs_amount', 'mean'),
        std_amount=('abs_amount', 'std'),
        txn_count=('TransactionId','nunique') if 'TransactionId' in d.columns else (date_col,'count'),
        provider_count=('ProviderId','nunique') if 'ProviderId' in d.columns else ('CustomerId','nunique'),
        product_count=('ProductId','nunique') if 'ProductId' in d.columns else ('CustomerId','nunique'),
        category_count=('ProductCategory','nunique') if 'ProductCategory' in d.columns else ('CustomerId','nunique'),
        channel_count=('ChannelId','nunique') if 'ChannelId' in d.columns else ('CustomerId','nunique'),
        fraud_rate=('FraudResult', 'mean') if 'FraudResult' in d.columns else (date_col, lambda s: 0.0)
    ).reset_index()

    # Hour features
    if date_col in d.columns:
        d['hour'] = d[date_col].dt.hour
        hr = d.groupby(id_col)['hour'].agg(['mean','std']).reset_index().rename(columns={'mean':'hour_mean','std':'hour_std'})
        agg = agg.merge(hr, left_on=id_col, right_on=id_col, how='left')

    return agg


@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load trained model from file if available."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model at {path}: {e}")
            return None
    return None


# ----------------- Scorecard Class -----------------
class Scorecard:
    """Simple credit score conversion based on PD."""
    def __init__(self, base_score=600.0, p_at_base=0.5, pdo=50.0):
        self.base_score = base_score
        self.odds0 = p_at_base / (1 - p_at_base)
        self.factor = pdo / np.log(2)

    def prob_to_score(self, p: float) -> float:
        p = np.clip(p, 1e-6, 1-1e-6)
        odds = (1 - p) / p
        return float(self.base_score + self.factor * np.log(odds / self.odds0))


scorecard = Scorecard()

# ----------------- Sidebar -----------------
st.sidebar.header("Controls & Inputs")
upload = st.sidebar.file_uploader("Upload transactions CSV", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample demo data (small)")
run_clustering = st.sidebar.button("Compute RFM & Cluster")
show_model_section = st.sidebar.checkbox("Enable model predictions")

# ----------------- Main Page -----------------
col1, col2 = st.columns([2,1])

# Left column: Data upload and display
with col1:
    if upload is not None:
        raw_df = load_csv(upload)
        st.success(f"Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]} columns")
        if st.checkbox("Show raw sample (first 5 rows)"):
            st.dataframe(raw_df.head())
    elif use_sample:
        # Generate synthetic demo data
        n = 1000
        rng = np.random.default_rng(42)
        custs = [f"C{int(x)}" for x in rng.integers(1, 200, size=n)]
        times = pd.date_range('2025-01-01', periods=n, freq='H')
        amounts = rng.normal(100, 50, size=n)
        raw_df = pd.DataFrame({
            'TransactionId': np.arange(n),
            'CustomerId': custs,
            'TransactionStartTime': times,
            'Value': amounts,
            'ProviderId': rng.integers(1,6,n),
            'ProductId': rng.integers(1,20,n),
            'ProductCategory': rng.integers(1,8,n),
            'ChannelId': rng.choice(['web','ios','android'], size=n),
            'FraudResult': rng.choice([0,0,0,1], size=n)
        })
        st.info("Using generated demo dataset")
        if st.checkbox("Show demo sample (first 5 rows)"):
            st.dataframe(raw_df.head())
    else:
        st.info("Upload a dataset or select 'Use sample demo data'.")
        raw_df = None

# Right column: Model status
with col2:
    st.markdown("### Model status")
    model = None
    if show_model_section:
        model = load_model()
        if model:
            st.success(f"Model loaded from {MODEL_PATH}")
            st.markdown("Predicts probabilities: available" if hasattr(model, 'predict_proba') else
                        "Model does not support predict_proba — probability mode unavailable")
        else:
            st.warning(f"No model found at {MODEL_PATH}")
    else:
        st.write("Model predictions disabled")

# -----------------
