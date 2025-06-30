# tests/test_data_processing.py

import pandas as pd
from src.data_processing import extract_datetime_features, create_aggregate_features

def test_extract_datetime_features():
    df = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 10:00:00', '2025-01-02 14:30:00']
    })
    df = extract_datetime_features(df)

    assert 'transaction_hour' in df.columns
    assert df.loc[0, 'transaction_hour'] == 10
    assert df.loc[1, 'transaction_day'] == 2

def test_create_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300]
    })
    df_agg = create_aggregate_features(df)

    assert 'Total_Transaction_Amount' in df_agg.columns
    assert df_agg[df_agg['CustomerId'] == 1]['Total_Transaction_Amount'].values[0] == 300
    assert df_agg[df_agg['CustomerId'] == 2]['avg_transaction_amount'].values[0] == 300
