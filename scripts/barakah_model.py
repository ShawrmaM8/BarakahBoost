from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from utils.io_utils import ROOT, ensure_dirs, save_json


def train_and_analyze() -> dict:
    """
    Train a Random Forest model to predict spiritual outcomes based on activities.

    Returns:
        dict: Dictionary containing model performance metrics and feature importance
    """
    ensure_dirs()

    # Load processed data
    path = os.path.join(ROOT, "data", "processed", "daily_features.csv")
    if not os.path.exists(path):
        return {"status": "no_data"}

    df = pd.read_csv(path)

    # Calculate basic correlations
    corr = df.corr(numeric_only=True).fillna(0)

    # Create aggregate outcome measure
    df["avg_outcome"] = df[["clarity", "focus", "calm", "productivity"]].mean(axis=1)
    df = df.dropna(subset=["avg_outcome"])  # Keep only rows with outcomes

    # Check if we have enough data for ML
    if len(df) < 10:
        result = {
            "status": "insufficient_data",
            "correlations": corr["avg_outcome"].to_dict()
        }
        save_results(result)
        return result

    # Prepare features and target
    X, y = prepare_features_and_target(df)

    # Train and evaluate model
    model, scores = train_model(X, y)

    # Extract feature importances
    importances = dict(zip(X.columns, model.feature_importances_.round(6)))

    # Compile results
    result = {
        "status": "ok",
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "feature_importances": importances,
        "correlations_with_outcome": corr["avg_outcome"].to_dict(),
        "n_samples": len(df)
    }

    # Save results and model
    save_results(result)
    save_model(model)

    return result


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare feature matrix and target vector for modeling.

    Args:
        df: Input DataFrame with all features and outcomes

    Returns:
        tuple: Feature matrix (X) and target vector (y)
    """
    feature_columns = [
        "prayer_on_time", "quran_items", "dhikr_reps", "sadaqah_amount",
        "sleep_hours", "prod_minutes", "dist_minutes", "other_good", "other_bad"
    ]

    # Ensure all required columns exist, fill missing with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns].fillna(0)
    y = df["avg_outcome"].values

    return X, y


def train_model(X: pd.DataFrame, y: np.ndarray) -> tuple[RandomForestRegressor, np.ndarray]:
    """
    Train a Random Forest model and perform cross-validation.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        tuple: Trained model and cross-validation scores
    """
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_split=5,  # Added to prevent overfitting with small datasets
        max_depth=10  # Added to prevent overfitting
    )

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="r2")

    # Train final model on all data
    model.fit(X, y)

    return model, scores


def save_results(results: dict) -> None:
    """
    Save model results to JSON file.

    Args:
        results: Dictionary containing model results
    """
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    results_path = os.path.join(ROOT, "models", "feature_importances.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def save_model(model: RandomForestRegressor) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained Random Forest model
    """
    try:
        import joblib
    except ImportError:
        try:
            from sklearn.externals import joblib  # Fallback for old sklearn versions
        except ImportError:
            print("Warning: joblib not available, model not saved")
            return

    model_path = os.path.join(ROOT, "models", "rf_outcome_model.pkl")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    try:
        results = train_and_analyze()
        print("Model training completed successfully!")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during model training: {e}")
        # Save error information for debugging
        error_result = {
            "status": "error",
            "error_message": str(e)
        }
        save_results(error_result)