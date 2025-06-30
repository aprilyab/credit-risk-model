# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def create_aggregate_features(df):
    grouped = df.groupby("CustomerId")["Amount"]
    df_agg = pd.DataFrame({'CustomerId': grouped.groups.keys()})
    df_agg.set_index('CustomerId', inplace=True)
    df_agg["Total_Transaction_Amount"] = grouped.sum()
    df_agg["avg_transaction_amount"] = grouped.mean()
    df_agg["transaction_count"] = grouped.count()
    df_agg["std_transaction_amount"] = grouped.std()
    return df_agg.reset_index()

def extract_datetime_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df

def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(columns))
    df = pd.concat([df.drop(columns=columns).reset_index(drop=True), encoded_df], axis=1)
    return df

def impute_missing(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])
    num_imputer = SimpleImputer(strategy="median")
    for column in numerical_columns.columns:
        df[column] = num_imputer.fit_transform(df[[column]]).ravel()
    return df

def normalize(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])
    normalizer = MinMaxScaler()
    for column in numerical_columns.columns:
        df[column] = normalizer.fit_transform(df[[column]])
    return df

def standardize(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    for column in numerical_columns.columns:
        df[column] = scaler.fit_transform(df[[column]])
    return df
