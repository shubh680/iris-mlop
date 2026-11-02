"""Unit tests for model training and evaluation metrics."""

import json
from pathlib import Path
from src.train import train_model
from src.utils import load_iris_regression_data, get_train_test_data, FEATURE_NAMES, TARGET_NAME


def test_load_iris_data():
    """Verify that dataset loads with correct columns and non-empty rows."""
    df = load_iris_regression_data()
    assert not df.empty
    assert len(df) == 150
    for feat in FEATURE_NAMES:
        assert feat in df.columns
    assert TARGET_NAME in df.columns
    assert "species" in df.columns


def test_train_test_split():
    """Verify train-test split dimensions."""
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.2, random_state=42)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert list(X_train.columns) == FEATURE_NAMES
    assert len(y_train) == 120


def test_model_training_and_metrics_quality(tmp_path: Path):
    """Test model training pipeline and verify quality gate."""
    metrics = train_model(model_dir=tmp_path, random_state=42)

    model_file = tmp_path / "linear_regression_model.joblib"
    metrics_file = tmp_path / "metrics.json"

    assert model_file.exists()
    assert metrics_file.exists()

    # Quality Gate assertions
    assert metrics["metrics"]["test_r2"] > 0.85, "Model R² must meet quality threshold"
    assert metrics["metrics"]["test_rmse"] < 1.0, "Model RMSE must be within reasonable bounds"
    assert metrics["metrics"]["test_mae"] < 1.0

    # Ensure json is valid
    with open(metrics_file, "r") as f:
        loaded = json.load(f)
    assert "metrics" in loaded
    assert "model_parameters" in loaded
