import pandas as pd
from utils import load_data
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Convert Date and Time to datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Set DateTime as index
    df = df.set_index('DateTime')
    
    # Handle missing values (forward fill for time series)
    df = df.fillna(method='ffill')
    
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

if __name__ == "__main__":
    # Load data
    df = load_data('data/household_power_consumption.txt')
    
    # Preprocess data
    processed_df, scaler = preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv('data/processed_data.csv')
    print("Data preprocessing completed and saved.")