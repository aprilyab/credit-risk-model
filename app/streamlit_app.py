"""
Credit Risk Proxy Dashboard
---------------------------
Run with:
    streamlit run app.py
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
import seaborn as sns
import os

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------- Constants -----------------
MODEL_PATH = "models/best_model.pkl"

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 Credit Risk — RFM Proxy & Dashboard")

st.markdown(
    """
    This dashboard helps analyze customer credit risk using:
    - **RFM Analysis** (Recency, Frequency, Monetary)
    - **KMeans clustering** to segment customers
    - **Optional ML model predictions** if a trained model is provided
    - **Credit score conversion** and **SHAP explainability**
    """
)

# ----------------- Utility Functions -----------------
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")

@st.cache_data
def compute_rfm(df: pd.DataFrame, id_col="CustomerId", date_col="TransactionStartTime",
                amount_cols=("Value", "Amount")) -> pd.DataFrame:
    d = df.copy()
    # Amount column
    amount_col = next((c for c in amount_cols if c in d.columns), None)
    if amount_col is None:
        raise ValueError(f"No amount column found among {amount_cols}")

    # Parse datetime
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    snapshot = d[date_col].max() + pd.Timedelta(days=1)
    d["monetary_abs"] = d[amount_col].abs()

    # RFM
    grp = d.groupby(id_col).agg(
        Recency=(date_col, lambda s: (snapshot - s.max()).days),
        Frequency=("TransactionId", "nunique") if "TransactionId" in d.columns else (date_col, "count"),
        Monetary=("monetary_abs", "sum")
    ).reset_index()
    return grp

@st.cache_data
def label_high_risk_by_rfm(rfm_df: pd.DataFrame, random_state: int = 42):
    feats = ["Recency", "Frequency", "Monetary"]
    r = rfm_df.copy()
    r[feats] = r[feats].fillna(0.0)

    # Standardize & cluster
    Z = StandardScaler().fit_transform(r[feats])
    km = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    clusters = km.fit_predict(Z)
    r["cluster"] = clusters

    # Find high-risk cluster
    clust_stats = r.groupby("cluster")[feats].mean()
    score = (
        clust_stats["Recency"].rank(ascending=False) +
        clust_stats["Frequency"].rank(ascending=True) +
        clust_stats["Monetary"].rank(ascending=True)
    )
    high_risk_cluster = score.idxmax()
    r["is_high_risk"] = (r["cluster"] == high_risk_cluster).astype(int)

    return r, km

@st.cache_data
def aggregate_features_from_transactions(df: pd.DataFrame, id_col="CustomerId",
                                         date_col="TransactionStartTime") -> pd.DataFrame:
    d = df.copy()
    amount_col = "Value" if "Value" in d.columns else ("Amount" if "Amount" in d.columns else None)
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["abs_amount"] = d[amount_col].abs()

    agg = d.groupby(id_col).agg(
        total_amount=("abs_amount", "sum"),
        avg_amount=("abs_amount", "mean"),
        txn_count=("TransactionId","nunique") if "TransactionId" in d.columns else (date_col,"count"),
        provider_count=("ProviderId","nunique") if "ProviderId" in d.columns else ("CustomerId","nunique"),
        product_count=("ProductId","nunique") if "ProductId" in d.columns else ("CustomerId","nunique"),
        channel_count=("ChannelId","nunique") if "ChannelId" in d.columns else ("CustomerId","nunique"),
        fraud_rate=("FraudResult", "mean") if "FraudResult" in d.columns else (date_col, lambda s: 0.0)
    ).reset_index()
    return agg

@st.cache_resource
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model at {path}: {e}")
            return None
    return None

# ----------------- Scorecard -----------------
class Scorecard:
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
st.sidebar.header("⚙️ Controls")
upload = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample demo data (1000 rows)")
run_clustering = st.sidebar.button("Compute RFM & Cluster")
show_model_section = st.sidebar.checkbox("Enable model predictions")

# ----------------- Data Handling -----------------
if upload is not None:
    raw_df = load_csv(upload)
    st.success(f"✅ Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]} columns")
elif use_sample:
    rng = np.random.default_rng(42)
    n = 1000
    raw_df = pd.DataFrame({
        "TransactionId": np.arange(n),
        "CustomerId": rng.choice([f"C{i}" for i in range(1,200)], size=n),
        "TransactionStartTime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "Value": rng.normal(100, 50, size=n),
        "ProviderId": rng.integers(1,6,n),
        "ProductId": rng.integers(1,20,n),
        "ProductCategory": rng.integers(1,8,n),
        "ChannelId": rng.choice(["web","ios","android"], size=n),
        "FraudResult": rng.choice([0,0,0,1], size=n)
    })
    st.info("ℹ️ Using synthetic demo dataset")
else:
    st.warning("Upload a dataset or select 'Use sample demo data'")
    raw_df = None

# ----------------- Main Analysis -----------------
if raw_df is not None and run_clustering:
    st.subheader("📌 RFM Analysis & Clustering")
    rfm_df = compute_rfm(raw_df)
    rfm_labeled, km_model = label_high_risk_by_rfm(rfm_df)

    st.dataframe(rfm_labeled.head())

    # Cluster distributions
    st.markdown("### Cluster Summary")
    st.dataframe(rfm_labeled.groupby("cluster")[["Recency","Frequency","Monetary"]].mean())

    # PCA plot
    feats = ["Recency","Frequency","Monetary"]
    Z = StandardScaler().fit_transform(rfm_labeled[feats])
    pca = PCA(n_components=2)
    pc = pca.fit_transform(Z)
    fig, ax = plt.subplots()
    sns.scatterplot(x=pc[:,0], y=pc[:,1], hue=rfm_labeled["cluster"], palette="Set2", ax=ax)
    st.pyplot(fig)

# ----------------- Model Predictions -----------------
if raw_df is not None and show_model_section:
    st.subheader("🤖 Model Predictions")
    model = load_model()
    if model is None:
        st.error("No model found at models/best_model.pkl")
    else:
        agg = aggregate_features_from_transactions(raw_df)
        if hasattr(model, "predict_proba"):
            agg["PD"] = model.predict_proba(agg.drop(columns=["CustomerId"]))[:,1]
            agg["Score"] = agg["PD"].apply(scorecard.prob_to_score)
            st.dataframe(agg.head())

            # PD Histogram
            fig, ax = plt.subplots()
            sns.histplot(agg["PD"], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Model does not support predict_proba")

        # SHAP Explainability
        if HAS_SHAP and hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model, agg.drop(columns=["CustomerId","PD","Score"]))
            shap_values = explainer(agg.drop(columns=["CustomerId","PD","Score"]))
            st.subheader("🔎 SHAP Explainability")
            st.pyplot(shap.plots.beeswarm(shap_values, show=False))

# ----------------- Single-Customer Prediction -----------------
if raw_df is not None and show_model_section:
    st.sidebar.subheader("🔮 Single Customer Prediction")
    cust_id = st.sidebar.selectbox("Choose Customer ID", raw_df["CustomerId"].unique())
    if st.sidebar.button("Predict Customer Risk"):
        agg = aggregate_features_from_transactions(raw_df)
        row = agg[agg["CustomerId"] == cust_id]
        if not row.empty and model is not None and hasattr(model, "predict_proba"):
            pd_prob = model.predict_proba(row.drop(columns=["CustomerId"]))[:,1][0]
            score = scorecard.prob_to_score(pd_prob)
            st.success(f"Customer {cust_id} → PD={pd_prob:.2%}, Score={score:.0f}")
