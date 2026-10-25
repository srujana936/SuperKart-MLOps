
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Product_Store_Sales_Total"

CATEGORICAL_COLUMNS = [
    "Product_Id",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

NUMERIC_COLUMNS = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

def build_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "superkart_train.csv"
    test_path = DATA_DIR / "superkart_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training/test files are missing. Run prepare_data.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Candidate models and modest parameter grids for practical runtime.
    candidates = {
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 5]
            }
        ),
        "RandomForest": (
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 3]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3]
            }
        )
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fitted_models = {}
    comparison = []

    for model_name, (estimator, param_grid) in candidates.items():
        print(f"\nTraining and tuning: {model_name}")

        pipeline = build_pipeline(estimator)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1
        )

        search.fit(X_train, y_train)

        fitted_models[model_name] = search.best_estimator_
        comparison.append({
            "model": model_name,
            "cv_rmse": float(-search.best_score_),
            "best_parameters": search.best_params_
        })

        print("Best cross-validation RMSE:", round(-search.best_score_, 4))
        print("Best parameters:", search.best_params_)

    # Select using training-set cross-validation, not test-set performance.
    comparison.sort(key=lambda row: row["cv_rmse"])
    best_name = comparison[0]["model"]
    best_model = fitted_models[best_name]

    # Evaluate the selected model on the held-out test set once.
    predictions = best_model.predict(X_test)

    test_metrics = {
        "model_name": best_name,
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target": TARGET
    }

    # Save the trained model artifact.
    model_path = MODEL_DIR / "superkart_best_model.joblib"
    joblib.dump(best_model, model_path, compress=3)

    # Save metrics and model comparison for reproducibility.
    report = {
        "selected_model": best_name,
        "selection_method": "lowest 5-fold cross-validation RMSE on training data",
        "candidate_models": comparison,
        "final_test_metrics": test_metrics
    }

    with open(REPORT_DIR / "metrics.json", "w") as file:
        json.dump(report, file, indent=2)

    # Create a schema for the Streamlit form.
    schema = {
        "target": TARGET,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_defaults": {},
        "categorical_defaults": {}
    }

    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(X_train[col], errors="coerce")
        median = values.median()
        schema["numeric_defaults"][col] = (
            float(median) if pd.notna(median) else 0.0
        )

    for col in CATEGORICAL_COLUMNS:
        modes = X_train[col].dropna().astype(str).mode()
        schema["categorical_defaults"][col] = (
            modes.iloc[0] if not modes.empty else ""
        )

    with open(MODEL_DIR / "feature_schema.json", "w") as file:
        json.dump(schema, file, indent=2)

    print("\nModel comparison:")
    print(pd.DataFrame(comparison)[["model", "cv_rmse"]].to_string(index=False))

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    print("\nSaved model:", model_path)
    print("Saved schema:", MODEL_DIR / "feature_schema.json")
    print("Saved metrics:", REPORT_DIR / "metrics.json")

if __name__ == "__main__":
    main()
