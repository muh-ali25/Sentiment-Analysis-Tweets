# src/models/lf.py
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

def load_dataset(file_path: str):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def separate_features_and_target(data: pd.DataFrame, target_column: str):
    """
    Separate features and target variable from the dataset.
    """
    X = data.drop(columns=[target_column])
    y=data[target_column]
    return X,y

def train_test_split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model using Mean Squared Error and R-squared metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def save_model(model_directory: str, model_name:str, model:object):
    """
    Function to save the model.
    
    :param model_directory: Directory to save the model.
    :param model_name: Name of the model file.
    :param model: Model object.
    """
    try:
        os.makedirs(model_directory, exist_ok=True)
        model_path = os.path.join(model_directory, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print("Error in Saving Model File\t", e)
        raise e
    
def load_model(model_directory: str, model_name:str):
    """
    Function to load the model.
    
    :param model_directory: Directory of the saved model.
    :param model_name: Name of the saved model file.
    """
    try:
        model_path = os.path.join(model_directory, f"{model_name}.pkl") 
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except Exception as e:
        print("Error in Loading Model File\t", e)
        return None

def main(file_path: str, target_column: str, processed_columns: list = None):
    """
    Main function to load data, train and evaluate the model.
    
    :param file_path: Path to the dataset CSV file.
    :param target_column: Name of the target column.
    """
    df = load_dataset(file_path)
    df = df[processed_columns] if processed_columns else df
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    X, y = separate_features_and_target(df, target_column)
    
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)
    
    model = train_model(X_train, y_train)
    
    rmse, r2 = evaluate_model(model, X_test, y_test)
    print("Model Evaluation:")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Training Score: {model.score(X_train, y_train)}")
    print(f"RMSE: {rmse}")
    print(f"R^2 Score: {r2}")

    print("Saving Model: ")
   
    save_model(model_directory= "artifacts", model_name="linear_regression", model= model)
    
