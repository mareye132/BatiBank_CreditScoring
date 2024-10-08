# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def create_aggregate_features(data):
    # Create aggregate features by CustomerId
    aggregate_features = data.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('Amount', 'size'),
        Std_Dev_Transaction_Amount=('Amount', 'std')
    ).reset_index()
    
    return aggregate_features


def extract_features(data):
    # Extract time-based features from 'TransactionStartTime'
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year
    return data

def encode_categorical_variables(data):
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['ProductCategory'], drop_first=True)  # Replace 'ProductCategory' with actual categorical column name
    
    # Label Encoding
    label_encoder = LabelEncoder()
    data['FraudResult'] = label_encoder.fit_transform(data['FraudResult'])  # Replace 'FraudResult' with actual categorical column name
    
    return data

def handle_missing_values(data):
    # Imputation
    imputer = SimpleImputer(strategy='mean')  # Can also use median, mode, or other strategies
    data[['Amount']] = imputer.fit_transform(data[['Amount']])  # Replace 'Amount' with actual column with missing values
    
    # Removal
    data.dropna(axis=0, how='any', inplace=True)  # Removes any rows with missing values (adjust based on your needs)
    
    return data

def normalize_standardize_features(data):
    # Standardization
    standard_scaler = StandardScaler()
    
    # Apply scaling only to the columns that need it
    data[['Total_Transaction_Amount', 'Average_Transaction_Amount', 'Std_Dev_Transaction_Amount']] = standard_scaler.fit_transform(
        data[['Total_Transaction_Amount', 'Average_Transaction_Amount', 'Std_Dev_Transaction_Amount']]
    )
    
    return data

# Main function to call all feature engineering steps
def main(data):
    # Step 1: Create aggregate features
    aggregate_features = create_aggregate_features(data)
    
    # Step 2: Merge aggregate features back into the original dataset
    data = pd.merge(data, aggregate_features, on='CustomerId', how='left')
    
    # Step 3: Extract additional features from time-based columns
    data = extract_features(data)
    
    # Step 4: Normalize/Standardize the necessary features
    data = normalize_standardize_features(data)
    
    return data, aggregate_features
