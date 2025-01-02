import os
import io
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

import pandas as pd
import optuna
import joblib

from app.utils import evaluate_model, save_model


# Initialize FastAPI app
app = FastAPI()

# Setting up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define available models
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
    """Home page."""
    preprocessor_path = "static/models/label_encoder.pkl"

    # Delete existing encoder file if exists
    if os.path.exists(preprocessor_path):
        os.remove(preprocessor_path)

    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)  # "regression" or "classification"
):
    """Train models for regression or classification."""
    try:
        # Load data
        file_content = await file.read()
        data = pd.read_csv(io.StringIO(file_content.decode("utf-8-sig"))).dropna()

        # Select models based on task type
        models = regression_models if task_type == "regression" else classification_models

        # Encode target for classification
        label_encoder = None
        if task_type == "classification" and data[target_column].dtype == 'object':
            label_encoder = LabelEncoder()
            data[target_column] = label_encoder.fit_transform(data[target_column])

        # Split data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessor setup
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ],
            remainder="passthrough"
        )
        preprocessor_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        preprocessor_pipeline.fit(X_train)
        X_train = preprocessor_pipeline.transform(X_train)
        X_test = preprocessor_pipeline.transform(X_test)

        # Save preprocessor
        preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
        joblib.dump(preprocessor_pipeline, preprocessor_path)

        # Model training and evaluation
        best_model = None
        best_score = float('-inf') if task_type == "classification" else float('inf')
        best_model_name = ""

        for model_name, model in models.items():
            # Define Optuna objective
            def objective(trial):
                if "Random Forest" in model_name:
                    model.set_params(
                        n_estimators=trial.suggest_int('n_estimators', 50, 500),
                        max_depth=trial.suggest_int('max_depth', 3, 15),
                    )
                elif "XGBoost" in model_name:
                    model.set_params(
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        max_depth=trial.suggest_int('max_depth', 3, 15),
                        n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    )
                model.fit(X_train, y_train)
                return evaluate_model(model, X_test, y_test, task_type)

            study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize")
            study.optimize(objective, n_trials=20)

            # Train the model with best parameters
            model.set_params(**study.best_params)
            model.fit(X_train, y_train)
            score = evaluate_model(model, X_test, y_test, task_type)

            if (task_type == "regression" and score < best_score) or (task_type == "classification" and score > best_score):
                best_score = score
                best_model = model
                best_model_name = model_name

        # Save the best model
        save_model(best_model, "best_model")
        if label_encoder:
            joblib.dump(label_encoder, "static/models/label_encoder.pkl")

        return templates.TemplateResponse("train_result.html", {
            "request": request,
            "best_model_name": best_model_name,
            "best_score": best_score,
            "task_type": task_type
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_model/{task_type}")
async def download_model(task_type: str):
    """Download trained model."""
    model_path = "static/models/best_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(model_path, media_type="application/octet-stream", filename="best_model.pkl")


@app.get("/download_preprocessor/{task_type}")
async def download_preprocessor(task_type: str):
    """Download preprocessor."""
    preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
    if not os.path.exists(preprocessor_path):
        raise HTTPException(status_code=404, detail="Preprocessor file not found.")
    return FileResponse(preprocessor_path, media_type="application/octet-stream", filename=f"preprocessor_{task_type}.pkl")


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Page for making predictions."""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    """Predict based on uploaded CSV file."""
    file_content = await file.read()
    file_content_decoded = file_content.decode("utf-8-sig")
    data = pd.read_csv(io.StringIO(file_content_decoded))

    preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
    if not os.path.exists(preprocessor_path):
        raise HTTPException(status_code=404, detail=f"{task_type.capitalize()} Preprocessor file not found.")
    preprocessor = joblib.load(preprocessor_path)

    X = data.drop(columns=[target_column])
    X_transformed = preprocessor.transform(X)

    model_path = "static/models/best_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")
    model = joblib.load(model_path)

    label_encoder_path = "static/models/label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None

    predictions = model.predict(X_transformed)
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    return templates.TemplateResponse("predict_result.html", {
        "request": request,
        "predictions": predictions.tolist()
    })


@app.get("/predict_single", response_class=HTMLResponse)
async def single_predict_page(request: Request):
    """Page for making a single prediction."""
    return templates.TemplateResponse("predict_single.html", {"request": request})

@app.post("/predict_single", response_class=HTMLResponse)
async def predict_single_via_form(
    request: Request,
    feature_names: str = Form(...), 
    feature_values: str = Form(...),
    task_type: str = Form(...)  # "regression" or "classification"
):
    """Predict for a single input."""
    try:
        # Parse feature names and values
        feature_names_list = feature_names.split(",")
        feature_values_list = feature_values.split(",")
        
        # Convert values to appropriate types (float if possible)
        def try_convert(value):
            try:
                return float(value) if value.replace('.', '', 1).isdigit() else value
            except ValueError:
                return value

        feature_values_list = [try_convert(value) for value in feature_values_list]

        # Create DataFrame for single input
        single_data = pd.DataFrame([dict(zip(feature_names_list, feature_values_list))])

        # Load the preprocessor
        preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
        preprocessor = joblib.load(preprocessor_path)
        single_data_transformed = preprocessor.transform(single_data)

        # Load the best model
        model = joblib.load("static/models/best_model.pkl")

        # Make prediction
        prediction = model.predict(single_data_transformed)
        if task_type == "classification":
            label_encoder = joblib.load("static/models/label_encoder.pkl")
            prediction = label_encoder.inverse_transform(prediction)

        return templates.TemplateResponse("predict_single_result.html", {
        "request": request,
        "prediction": prediction[0]
    })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
