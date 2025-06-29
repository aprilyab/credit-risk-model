# Proxy Target Variable Engineering 

# Calculate RFM Metrics
import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\user\Desktop\credit-risk-model\data\processed\FE_cleaned_data.csv")
print(df.columns)

df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

# prepare snapshote date for calculating the recency
snapshot_date=df["TransactionStartTime"].max()+pd.Timedelta(days=1)
print(snapshot_date)

# calculate the recency
recency_df=df.groupby("CustomerId")["TransactionStartTime"].max().reset_index()
recency_df["recency"]=(snapshot_date-recency_df["TransactionStartTime"]).dt.days

# calculate the frequency
frequency_df=df.groupby("CustomerId")["TransactionId"].nunique().reset_index()
frequency_df.columns=["CustomerId","frequency"]

# calculate the monetary
monetary_df=df.groupby("CustomerId")["Amount"].sum().reset_index()
monetary_df.columns=["CustomerId","monetary"]

# Merge All RFM Metrics
rfm = recency_df.merge(frequency_df, on='CustomerId').merge(monetary_df, on='CustomerId')
print(rfm.head())




