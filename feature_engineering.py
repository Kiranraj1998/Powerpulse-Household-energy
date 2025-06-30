import pandas as pd
from utils import load_processed_data

def create_features(df):
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Lag features
    df['global_active_power_lag1'] = df['Global_active_power'].shift(1)
    df['global_active_power_lag24'] = df['Global_active_power'].shift(24)
    
    # Rolling statistics
    df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['Global_active_power'].rolling(window=24).std()
    
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Load processed data
    df = load_processed_data('data/processed_data.csv')
    
    # Create features
    feature_df = create_features(df)
    
    # Save data with features
    feature_df.to_csv('data/data_with_features.csv')
    print("Feature engineering completed and saved.")