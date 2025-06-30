import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def explore_data(df):
    # Basic information
    print("Dataset Info:")
    print(df.info())
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Plot distributions
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols].hist(bins=50, figsize=(20,15))
    plt.tight_layout()
    plt.savefig('data_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Load data
    df = load_data('data/household_power_consumption.txt')
    
    # Explore data
    explore_data(df)