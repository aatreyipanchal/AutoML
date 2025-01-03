import os, sys
import shutil
import io

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

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

    preprocessor_path = "static/models/label_encoder.pkl"

    # Check if the file exists
    if os.path.exists(preprocessor_path):
        # If the file exists, delete it
        os.remove(preprocessor_path)
        print(f"File {preprocessor_path} has been deleted.")
    else:
        print(f"File {preprocessor_path} does not exist.")
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    """Page for training model."""
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)  # "regression" or "classification"
):
    """Trains models dynamically for regression or classification."""
    # Load data
    file_content = await file.read()
    file_content_decoded = file_content.decode("utf-8-sig")
    data = pd.read_csv(io.StringIO(file_content_decoded))
    data_cleaned = data.dropna()

    # Determine task type models
    models = regression_models if task_type == "regression" else classification_models
    model_names = list(models.keys())  # List of all model names

    # Initialize LabelEncoder for classification tasks
    label_encoder = None
    if task_type == "classification" and data_cleaned[target_column].dtype == 'object':
        label_encoder = LabelEncoder()
        data_cleaned[target_column] = label_encoder.fit_transform(data_cleaned[target_column])

    # Split data into features and target variable
    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Determine numeric and categorical features
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Create a preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    ) if task_type == "regression" else ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"  # Keep numeric columns as-is for classification
    )

    preprocessor_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    preprocessor_pipeline.fit(X_train)
    X_train = preprocessor_pipeline.transform(X_train)
    X_test = preprocessor_pipeline.transform(X_test)

    # Save preprocessor for future use
    preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
    joblib.dump(preprocessor_pipeline, preprocessor_path)

    # Train and optimize models
    best_model = None
    best_score = float('-inf') if task_type == "classification" else float('inf')
    best_model_name = ""
    model_scores = {}

    for model_name, model in models.items():
        # Optuna for hyperparameter optimization
        def objective(trial):
            if "Random Forest" in model_name:
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                model.set_params(n_estimators=n_estimators, max_depth=max_depth)
            elif "XGBoost" in model_name:
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                model.set_params(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

            model.fit(X_train, y_train)
            score = evaluate_model(model, X_test, y_test, task_type)
            return score

        study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize")
        study.optimize(objective, n_trials=20)

        model.set_params(**study.best_params)
        model.fit(X_train, y_train)

        score = evaluate_model(model, X_test, y_test, task_type)
        model_scores[model_name] = score

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
        "model_names": model_names,
        "model_scores": model_scores,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "task_type": task_type
    })


@app.get("/download_model/{task_type}")
async def download_model(task_type: str):
    model_path = "static/models/best_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(model_path, media_type='application/octet-stream', filename="best_model.pkl")

@app.get("/download_preprocessor/{task_type}")
async def download_preprocessor(task_type: str):
    preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
    if not os.path.exists(preprocessor_path):
        raise HTTPException(status_code=404, detail="Preprocessor file not found.")
    return FileResponse(preprocessor_path, media_type='application/octet-stream', filename=f"preprocessor_{task_type}.pkl")

@app.get("/save_models/{task_type}")
async def save_model(task_type: str, request: Request):
    return templates.TemplateResponse("save_model.html", {"request": request, "task_type": task_type} )

# @app.post("/save_models/{task_type}")
# async def save_model_local(task_type: str, request: Request, preprocessor_name: str = Form(...),
#     model_name: str = Form(...)):
    
#     task_type = task_type.strip()

#     # preprocessor_name: str = Form(...),
#     # model_name: str = Form(...),
    
#     # preprocessor_path = f"static/models/preprocessor_{task_type}.pkl"
#     # model_path = "static/models/best_model.pkl"
    
#      # Use absolute paths to avoid current working directory issues
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     preprocessor_path = os.path.join(base_dir, f"F:/Projects/AutoML/static/models/preprocessor_{task_type}.pkl")
#     model_path = os.path.join(base_dir, "F:/Projects/AutoML/static/models/best_model.pkl")
    
#     artifacts_folder = os.path.join(base_dir, "..\\artifacts")
#     os.makedirs(artifacts_folder, exist_ok=True)   # Ensure the folder exists
    
#     preprocessor_dest_path = os.path.join(artifacts_folder, f"{preprocessor_name}.pkl")
#     model_dest_path = os.path.join(artifacts_folder, f"{model_name}.pkl")
    
#     # C:\office\tasks\AutoML\static\models\best_model.pkl
    
#     try:
#         # Copy and rename the preprocessor model
#         shutil.copy(preprocessor_path, preprocessor_dest_path)

#         # Copy and rename the machine learning model
#         shutil.copy(model_path, model_dest_path)

#         # return JSONResponse(
#         #     content={
#         #         "message": "Models successfully saved.",
#         #         "preprocessor_path": preprocessor_dest_path,
#         #         "model_path": model_dest_path
#         #     },
#         #     status_code=200
#         # )
        
#         return templates.TemplateResponse("save_model.html", {"request": request, "task_type": task_type, "message": "Models successfully saved."})

#     except FileNotFoundError as e:
#         return JSONResponse(
#             content={"error": f"File not found: {e}"},
#             status_code=404
#         )
#     except Exception as e:
#         return JSONResponse(
#             content={"error": f"An error occurred: {e}"},
#             status_code=500
#         )

@app.post("/save_models/{task_type}")
async def save_model_local(
    task_type: str, request: Request, preprocessor_name: str = Form(...), model_name: str = Form(...)
):
    task_type = task_type.strip()

    # Use absolute paths to avoid current working directory issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths dynamically based on task type
    preprocessor_path = os.path.join(base_dir, f"static/models/preprocessor_{task_type}.pkl")
    model_path = os.path.join(base_dir, "static/models/best_model.pkl")

    # Define the artifacts folder
    artifacts_folder = os.path.join(base_dir, "../artifacts")
    os.makedirs(artifacts_folder, exist_ok=True)  # Ensure the folder exists

    # Define the destination paths for saved models
    preprocessor_dest_path = os.path.join(artifacts_folder, f"{preprocessor_name}.pkl")
    model_dest_path = os.path.join(artifacts_folder, f"{model_name}.pkl")

    try:
        # Debugging paths
        print(f"Preprocessor path: {preprocessor_path}")
        print(f"Model path: {model_path}")
        print(f"Artifacts folder: {artifacts_folder}")

        # Check if files exist
        if not os.path.exists(preprocessor_path):
            return JSONResponse(content={"error": f"Preprocessor file not found: {preprocessor_path}"}, status_code=404)

        if not os.path.exists(model_path):
            return JSONResponse(content={"error": f"Model file not found: {model_path}"}, status_code=404)

        # Copy and rename the preprocessor model
        shutil.copy(preprocessor_path, preprocessor_dest_path)

        # Copy and rename the machine learning model
        shutil.copy(model_path, model_dest_path)

        return templates.TemplateResponse(
            "save_model.html",
            {
                "request": request, 
                "task_type": task_type, 
                "message": "Models successfully saved."
            }
        )

    except FileNotFoundError as e:
        return JSONResponse(
            content={"error": f"File not found: {e}"},
            status_code=404
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {e}"},
            status_code=500
        )


@app.get("/predict_select", response_class=HTMLResponse)
async def select_predictions(request:Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Page for making predictions."""
    return templates.TemplateResponse("batch_predictions.html", {"request": request})

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

    try:
        # Perform predictions
        predictions = model.predict(X_transformed)

        # Format predictions for output
        if task_type == "classification":
            predictions = [int(pred) for pred in predictions]
        elif task_type == "regression":
            predictions = [float(pred) for pred in predictions]

        # Add predictions to the original data
        data["Predictions"] = predictions

        # Save the predictions as a CSV
        output_csv_path = "static/output/predictions.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        data.to_csv(output_csv_path, index=False)

        return templates.TemplateResponse(
            "predict_result.html", 
            {
                "request": request,
                "message": "Predictions generated successfully.",
                "download_link": f"/static/output/predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while predicting: {e}")



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
    
    
# @app.get("/all_models", response_class=HTMLResponse)
# async def all_models(request: Request):
#     return templates.TemplateResponse("all_models.html", {
#         "request": request
#     })
    
    
@app.get("/all_models", response_class=HTMLResponse)
async def all_models_post(request: Request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the artifacts folder
    artifacts_folder = os.path.join(base_dir, "..\\artifacts")
    
    # Ensure the folder exists
    if not os.path.exists(artifacts_folder):
        return templates.TemplateResponse("all_models.html", {
            "request": request,
            "preprocessor_models": [],
            "ml_models": [],
            "message": "Artifacts folder not found."
        })
    
    # List files in the artifacts folder
    all_models = os.listdir(artifacts_folder)
    
    # Separate models into preprocessors and ML models based on the name
    preprocessor_models = [model for model in all_models if "preprocessor" in model.lower()]
    ml_models = [model for model in all_models if "preprocessor" not in model.lower()]
    
    # Render the template with the models
    return templates.TemplateResponse("all_models.html", {
        "request": request,
        "preprocessor_models": preprocessor_models,
        "ml_models": ml_models,
        "message": None if all_models else "No models found in the artifacts folder."
    })
    