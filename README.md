# Powerpulse-Household-energy

A machine learning pipeline for predicting household energy consumption using historical data. This system helps consumers understand usage patterns and enables providers to forecast demand.

## 🌟 Features
- End-to-end ML pipeline from raw data to predictions
- Multiple model comparison (XGBoost, Random Forest, etc.)
- Feature engineering for temporal patterns
- Comprehensive evaluation metrics
- Visualizations of key insights

## 📂 Project Structure

energy-consumption/
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Cleaned and normalized data
│ └── features/ # Data with engineered features
├── models/ # Trained model binaries
├── reports/ # Evaluation reports
├── visualizations/ # Generated plots and charts
└── scripts/
├── 1_data_understanding.py
├── 2_data_preprocessing.py
├── 3_feature_engineering.py
├── 4_model_training.py
├── 5_model_evaluation.py
└── utils.py # Helper functions

Usage
Run the pipeline sequentially:

# 1. Exploratory Data Analysis
python scripts/1_data_understanding.py

# 2. Data Cleaning and Normalization
python scripts/2_data_preprocessing.py

# 3. Feature Engineering
python scripts/3_feature_engineering.py

# 4. Model Training
python scripts/4_model_training.py

# 5. Model Evaluation
python scripts/5_model_evaluation.py

📊 Results
Model Performance
Model	RMSE	MAE	R²
XGBoost	0.043	0.030	0.891
Random Forest	0.045	0.032	0.882
Linear Regression	0.062	0.048	0.774
