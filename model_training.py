import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import joblib

def train_models(X_train, y_train):
    """Train multiple regression models with optimized settings"""
    models = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Random Forest': RandomForestRegressor(
            n_estimators=50,       # Reduced from 100 for faster training
            max_depth=10,          # Limit tree depth
            n_jobs=-1,             # Use all CPU cores
            random_state=42
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=100,         # Faster alternative to standard GBM
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
            tree_method='hist'     # Faster training method
        ),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(50,),  # Simplified architecture
            max_iter=200,
            random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        try:
            model.fit(X_train, y_train)
            print(f"✓ Completed {name}")
            trained_models[name] = model
        except Exception as e:
            print(f"✗ Failed to train {name}: {str(e)}")
    
    return trained_models

def prepare_data(filepath):
    """Load and preprocess data"""
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert target to numeric
    y = pd.to_numeric(df['Global_active_power'], errors='coerce')
    
    # Select only numeric features
    X = df.select_dtypes(include=['number']).drop(
        ['Global_active_power'], 
        axis=1, 
        errors='ignore'
    )
    
    # Drop rows with missing values
    valid_idx = X.notnull().all(axis=1) & y.notnull()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    
    # Prepare data
    try:
        X, y = prepare_data('data/data_with_features.csv')
        print(f"\nData loaded successfully. Shape: {X.shape}")
        
        # Optionally subsample for faster testing
        # X = X.sample(frac=0.1, random_state=42)
        # y = y.loc[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        print("Data split completed")
        
        # Train models
        models = train_models(X_train, y_train)
        
        # Save models
        model_path = os.path.join('models', 'trained_models.pkl')
        joblib.dump(models, model_path)
        print(f"\nModels saved to {model_path}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")