"""Train Linear Regression model on Iris dataset and persist artifacts."""

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import get_train_test_data, FEATURE_NAMES, TARGET_NAME

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = DEFAULT_MODEL_DIR / "linear_regression_model.joblib"
METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"


def train_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Train linear regression model, calculate metrics, and save artifacts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = get_train_test_data(
        test_size=test_size, random_state=random_state
    )

    # 2. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Evaluate Model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    test_mse = float(mean_squared_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))

    # Coefficients summary
    coefficients = {
        feature: float(coef)
        for feature, coef in zip(FEATURE_NAMES, model.coef_)
    }
    intercept = float(model.intercept_)

    metrics = {
        "dataset": "Iris (scikit-learn)",
        "target": TARGET_NAME,
        "features": FEATURE_NAMES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_mse": round(test_mse, 4),
            "test_rmse": round(test_rmse, 4),
            "test_mae": round(test_mae, 4),
        },
        "model_parameters": {
            "intercept": round(intercept, 4),
            "coefficients": {k: round(v, 4) for k, v in coefficients.items()},
        },
    }

    # 4. Save artifacts
    model_file = model_dir / "linear_regression_model.joblib"
    metrics_file = model_dir / "metrics.json"

    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

    return metrics


if __name__ == "__main__":
    train_model()
