import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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


#  Encode Categorical Variables with One label encoding

# Find categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category'])

# columns with too many unique categories (e.g., >100)
low_cardinality_cols = [col for col in categorical_columns.columns if df[col].nunique() <= 100]

#	One-Hot Encoding
# initialize the encoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")

# fit and transform the data
encoded_array=encoder.fit_transform(df[low_cardinality_cols])

# Get new column names
encoded_df=pd.DataFrame(encoded_array,columns=encoder.get_feature_names_out(low_cardinality_cols))

# Concatenate with original DataFrame
df_encoded = pd.concat([df.drop(columns=low_cardinality_cols).reset_index(drop=True), encoded_df], axis=1)


#  Encode Categorical Variables with abel encoding

# initialize the encoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")

for column in low_cardinality_cols:
    df[column]=encoder.fit_transform(df[[column]])

#  Handle Missing Values using imputer
numerical_columns = df.select_dtypes(include=['int64', 'float64'])

# for numerical features with missing values
num_imputer=SimpleImputer(strategy="median")
for column in numerical_columns.columns:
    df[column]=num_imputer.fit_transform(df[[column]]).ravel()

# for categorical features with missing values
cat_imputer=SimpleImputer(strategy="most_frequent")
for column in categorical_columns.columns:
    df[column]=cat_imputer.fit_transform(df[[column]]).ravel()


# normalization of the data
normalizer=MinMaxScaler()
for column in numerical_columns.columns:
    df[column]=normalizer.fit_transform(df[[column]])

# standardaization of the data
standaizer=StandardScaler()
for column in numerical_columns.columns:
    df[column]=standaizer.fit_transform(df[[column]])
print(df.head())



