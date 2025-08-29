# src/models/rf.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def main(file_path, target_column, processed_columns=None):
    # Load data
    df = pd.read_csv(file_path)

    if processed_columns:
        df = df[processed_columns]
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train/test split (e.g. 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestRegressor(random_state=42, n_estimators=100)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regressor Results:\nR2 Score: {r2:.4f}\nRMSE: {rmse:.4f}")

    return r2, rmse
