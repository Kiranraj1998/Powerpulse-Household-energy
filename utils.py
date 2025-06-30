import pandas as pd

def load_data(filepath):
    """Load the raw data"""
    df = pd.read_csv(filepath, sep=';', na_values=['?'])
    return df

def load_processed_data(filepath):
    """Load processed data"""
    df = pd.read_csv(filepath, parse_dates=['DateTime'], index_col='DateTime')
    return df

def load_feature_data(filepath):
    """Load data with features"""
    df = pd.read_csv(filepath, parse_dates=['DateTime'], index_col='DateTime')
    return df