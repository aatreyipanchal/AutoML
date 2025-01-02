import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def preprocess_data(data: pd.DataFrame, target_column: str, task_type: str):
    """
    Preprocess the dataset with separate preprocessors for regression and classification:
    - Regression: Numerical data only (imputation + scaling).
    - Classification: Numerical (imputation + scaling) + Categorical (imputation + one-hot encoding) + Label Encoding for target if non-numeric.
    - Returns train-test splits for features and target variables.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

    # Define regression preprocessor
    regression_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_features),
        ]
    )

    # Define classification preprocessor
    classification_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    # Handle target variable for classification
    if task_type == "classification":
        # Encode non-numeric target values
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Save the LabelEncoder for later use
            if not os.path.exists("models"):
                os.makedirs("models")
            joblib.dump(label_encoder, "static/models/label_encoder.pkl")
    elif task_type == "regression":
        # Ensure the target is numeric for regression
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target variable must be numeric for regression tasks.")

    # Select the appropriate preprocessor
    preprocessor = regression_preprocessor if task_type == "regression" else classification_preprocessor

    # Split the data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Save the preprocessor
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(preprocessor, f"static/models/{task_type}_preprocessor.pkl")

    # Transform the features
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
    model_path = f"static/models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
