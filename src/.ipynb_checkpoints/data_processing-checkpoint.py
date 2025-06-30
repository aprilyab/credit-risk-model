import pandas as pd 
import numpy as np

#load the cleaned data
df=pd.read_csv(r"C:\Users\user\Desktop\credit-risk-model\data\processed\cleaned_data.csv")

# 	Create Aggregate Features
#   Group by CustomerId
grouped=df.groupby("CustomerId")["Amount"]

# Initialize an empty DataFrame with CustomerId as index
df_agg = pd.DataFrame({'CustomerId': grouped.groups.keys()})
df_agg.set_index('CustomerId', inplace=True)

# Insert each new column step-by-step
df_agg["Total_Transaction_Amount"]=grouped.sum()
df_agg["avg_transaction_amount"]=grouped.mean()
df_agg["transaction_count"]=grouped.count()
df_agg["std_transaction_amount"]=grouped.std()

# Reset the index to bring CustomerId back as a column
df_agg = df_agg.reset_index()

#   Extract DateTime Features from 
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['transaction_hour'] = df['TransactionStartTime'].dt.hour
df['transaction_day'] = df['TransactionStartTime'].dt.day
df['transaction_month'] = df['TransactionStartTime'].dt.month
df['transaction_year'] = df['TransactionStartTime'].dt.year


# 5 Encode Categorical Variables
#	One-Hot Encoding