import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(data: pd.DataFrame, target_column: str):
    """
    Preprocess the dataset:
    - Split into features (X) and target (y)
    - Apply preprocessing (scaling, encoding)
    - Return train-test splits.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Separate categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Define transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on training data
    preprocessor.fit(X_train)

    # Save the preprocessor
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Transform the data
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, task_type: str):
    """
    Evaluate the model based on task type (regression or classification):
    - Regression: Use Mean Squared Error (MSE).
    - Classification: Use Accuracy, Precision, Recall, and F1-score.
    """
    y_pred = model.predict(X_test)

    if task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")
        return mse
    elif task_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        return accuracy


def save_model(model, model_name: str):
    """
    Save the trained model as a .pkl file for reuse.
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
