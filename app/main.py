from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.utils import preprocess_data, evaluate_model, save_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import pandas as pd
import optuna
import joblib
import os
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

# Setting up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Models for regression and classification
regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "XGBoost Regressor": xgb.XGBRegressor(),
}

classification_models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "XGBoost Classifier": xgb.XGBClassifier(),
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Home page with options for regression and classification."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    """Page for training model."""
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train")
async def train_model(
    file: UploadFile = File(...), 
    target_column: str = Form(...),
    task_type: str = Form(...)  # "regression" or "classification"
):
    """Trains models dynamically for regression or classification."""
    # data = pd.read_csv(file.file, encoding="latin1")
    file_content = await file.read()
    file_content_decoded = file_content.decode("utf-8-sig")
    data = pd.read_csv(io.StringIO(file_content_decoded))
    data_cleaned = data.dropna()

    # Determine task type
    models = regression_models if task_type == "regression" else classification_models

    # Initialize LabelEncoder for classification tasks
    label_encoder = None
    if task_type == "classification" and data_cleaned[target_column].dtype == 'object':
        label_encoder = LabelEncoder()
        data_cleaned[target_column] = label_encoder.fit_transform(data_cleaned[target_column])

    # Split data into features and target variable
    X_train, X_test, y_train, y_test = preprocess_data(data_cleaned, target_column)

    best_model = None
    best_score = float('-inf') if task_type == "classification" else float('inf')
    best_model_name = ""

    for model_name, model in models.items():
        # Optuna for hyperparameter optimization
        def objective(trial):
            # Random Forest Hyperparameters
            if "Random Forest" in model_name:
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                model.set_params(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
            # XGBoost Hyperparameters
            elif "XGBoost" in model_name:
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                model.set_params(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

            model.fit(X_train, y_train)
            score = evaluate_model(model, X_test, y_test, task_type)
            return score

        # Optimize with Optuna
        study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize")
        study.optimize(objective, n_trials=20)

        # Train with best hyperparameters
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)

        score = evaluate_model(model, X_test, y_test, task_type)
        print(f"{model_name} - Score: {score}")

        # Update the best model based on the task type
        if (task_type == "regression" and score < best_score) or (task_type == "classification" and score > best_score):
            best_score = score
            best_model = model
            best_model_name = model_name

    # Save the best model and label encoder
    save_model(best_model, "best_model")
    if label_encoder:
        joblib.dump(label_encoder, "models/label_encoder.pkl")

    return {"message": f"Best model is {best_model_name} with score: {best_score}"}


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Page for making predictions."""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), target_column: str = Form(...)):
    data = pd.read_csv(file.file, encoding="latin1")

    preprocessor_path = "models/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        raise HTTPException(status_code=404, detail="Preprocessor file not found.")
    preprocessor = joblib.load(preprocessor_path)

    X = data.drop(columns=[target_column])
    X_transformed = preprocessor.transform(X)

    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")
    model = joblib.load(model_path)

    # Load the label encoder if it exists
    label_encoder_path = "models/label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None

    predictions = model.predict(X_transformed)
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    return {"predictions": predictions.tolist()}



@app.get("/predict_single", response_class=HTMLResponse)
async def single_predict_page(request: Request):
    """Page for making a single prediction."""
    return templates.TemplateResponse("predict_single.html", {"request": request})

@app.post("/predict_single")
async def predict_single_via_form(
    feature_names: str = Form(...), 
    feature_values: str = Form(...),
    task_type: str = Form(...)  # "regression" or "classification"
):
    """Predict for a single input."""
    try:
        # Parse features from form
        feature_names_list = feature_names.split(",")
        feature_values_list = feature_values.split(",")
        feature_values_list = [float(value) if value.replace('.', '', 1).isdigit() else value for value in feature_values_list]
        
        # Create DataFrame
        single_data = pd.DataFrame([dict(zip(feature_names_list, feature_values_list))])

        preprocessor_path = "models/preprocessor.pkl"
        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail="Preprocessor file not found.")
        preprocessor = joblib.load(preprocessor_path)

        single_transformed = preprocessor.transform(single_data)

        model_path = "models/best_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found.")
        model = joblib.load(model_path)

        prediction = model.predict(single_transformed)
        return {"prediction": float(prediction[0]) if task_type == "regression" else int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

