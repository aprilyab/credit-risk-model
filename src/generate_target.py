"""
Task 4 - Proxy Target Variable Engineering
-------------------------------------------
Since the dataset does not contain a direct "credit risk" label, we create a 
proxy target variable based on customer engagement. 

We use RFM (Recency, Frequency, Monetary) analysis + clustering 
to identify disengaged customers who are most likely to be "high risk".
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load transactions dataset
# (Assume `transactions` has at least these columns: CustomerId, TransactionDate, Amount)
# If your dataframe is named differently, replace accordingly.
transactions = pd.read_csv("data/processed/processed_data.csv", parse_dates=["TransactionStartTime"])

# 2. Define a snapshot date for recency calculation
snapshot_date = transactions["TransactionStartTime"].max() + pd.Timedelta(days=1)

# 3. Calculate RFM metrics
rfm = transactions.groupby("CustomerId").agg({
    "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,  # Recency
    "CustomerId": "count",                                       # Frequency
    "Amount": "sum"                                              # Monetary
}).rename(columns={
    "TransactionStartTime": "Recency",
    "CustomerId": "Frequency",
    "Amount": "Monetary"
}).reset_index()

# 4. Preprocess RFM features (scaling for clustering)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# 5. Apply KMeans clustering (3 segments)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# 6. Analyze clusters to identify high-risk group
# Typically: High Recency (long time since last purchase), Low Frequency, Low Monetary
cluster_profiles = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).reset_index()

# Determine the high-risk cluster
# Here: pick cluster with lowest Frequency & Monetary
high_risk_cluster = cluster_profiles.sort_values(["Frequency", "Monetary"]).iloc[0]["Cluster"]

# 7. Assign binary risk label
rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

# 8. Merge back into the main dataset
final_data = pd.merge(transactions, rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

# 9. Save processed dataset for modeling
final_data.to_csv("data/processed/final_dataset.csv", index=False)

print(" Proxy target variable 'is_high_risk' created and merged into dataset.")
print(rfm.head())
print(cluster_profiles)
