import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # Added this import

def evaluate_models(models, X_test, y_test):
    """Evaluate models and generate performance reports"""
    results = []
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            
            # Plot predictions vs actual
            plt.figure(figsize=(10,6))
            plt.scatter(y_test, y_pred, alpha=0.3)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{name} - Actual vs Predicted')
            
            # Save plots
            os.makedirs('model_results', exist_ok=True)
            plt.savefig(f'model_results/{name}_predictions.png')
            plt.close()
            
            print(f"✓ Evaluated {name}")
            
        except Exception as e:
            print(f"✗ Failed to evaluate {name}: {str(e)}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        # 1. Load data and models
        print("Loading data and models...")
        df = pd.read_csv('data/data_with_features.csv')
        models = joblib.load('models/trained_models.pkl')
        
        # 2. Prepare test data (must match training preprocessing)
        X = df.select_dtypes(include=['number']).drop('Global_active_power', axis=1, errors='ignore')
        y = pd.to_numeric(df['Global_active_power'], errors='coerce')
        
        # 3. Recreate the same test split used in training
        _, X_test, _, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            shuffle=False
        )
        
        # 4. Evaluate models
        print("\nEvaluating models...")
        results = evaluate_models(models, X_test, y_test)
        
        # 5. Save and display results
        results.to_csv('model_results/evaluation_results.csv', index=False)
        
        print("\nEvaluation Complete!")
        print(results.to_markdown())
        
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")