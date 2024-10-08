import pandas as pd
import numpy as np

def compute_rfms(data):
    # Convert 'TransactionStartTime' to timezone-naive datetime
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], utc=True).dt.tz_convert(None)
    
    # Recency
    recency_df = data.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (pd.Timestamp.now() - x.max()).days)
    ).reset_index()
    
    # Frequency
    frequency_df = data.groupby('CustomerId').agg(
        Frequency=('TransactionId', 'count')
    ).reset_index()
    
    # Monetary
    monetary_df = data.groupby('CustomerId').agg(
        Monetary=('Amount', 'sum')
    ).reset_index()

    # Merge the dataframes
    rfms_df = recency_df.merge(frequency_df, on='CustomerId').merge(monetary_df, on='CustomerId')
    
    return rfms_df


def classify_users_by_rfms(rfms_df):
    """
    Classify users as good or bad based on RFMS thresholds.
    :param rfms_df: DataFrame containing Recency, Frequency, and Monetary features.
    :return: DataFrame with Risk_Label (Good=0, Bad=1).
    """
    recency_threshold = rfms_df['Recency'].mean()
    frequency_threshold = rfms_df['Frequency'].mean()
    monetary_threshold = rfms_df['Monetary'].mean()
    
    rfms_df['Risk_Label'] = np.where(
        (rfms_df['Recency'] > recency_threshold) | 
        (rfms_df['Frequency'] < frequency_threshold) | 
        (rfms_df['Monetary'] < monetary_threshold),
        1,  # Bad
        0   # Good
    )
    
    return rfms_df
