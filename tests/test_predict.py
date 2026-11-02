"""Unit tests for inference and prediction validation."""

import pandas as pd
import pytest
from src.predict import IrisPredictor, predict_petal_length
from src.utils import FEATURE_NAMES


def test_predict_single_sample():
    """Test single sample prediction."""
    predictor = IrisPredictor()
    pred = predictor.predict(sepal_length=5.1, sepal_width=3.5, petal_width=0.2)

    assert isinstance(pred, float)
    assert pred > 0.0
    # Expected for setosa-like features: roughly around 1.0 - 2.0 cm
    assert 0.5 <= pred <= 3.0


def test_convenience_predict():
    """Test convenience function."""
    pred = predict_petal_length(sepal_length=6.0, sepal_width=3.0, petal_width=1.5)
    assert isinstance(pred, float)
    assert 2.0 <= pred <= 6.0


def test_predict_batch():
    """Test batch prediction."""
    predictor = IrisPredictor()
    batch_df = pd.DataFrame(
        [
            [5.0, 3.4, 0.2],
            [6.5, 3.0, 2.0],
        ],
        columns=FEATURE_NAMES,
    )
    preds = predictor.predict_batch(batch_df)
    assert len(preds) == 2
    assert all(isinstance(p, float) and p > 0.0 for p in preds)


def test_predict_invalid_inputs():
    """Test that non-positive inputs or non-numeric inputs raise ValueError."""
    predictor = IrisPredictor()
    with pytest.raises(ValueError, match="must be greater than 0"):
        predictor.predict(sepal_length=-1.0, sepal_width=3.0, petal_width=1.0)

    with pytest.raises(ValueError, match="must be numeric"):
        predictor.predict(sepal_length="invalid", sepal_width=3.0, petal_width=1.0)  # type: ignore


def test_predict_batch_missing_columns():
    """Test batch prediction with missing required column."""
    predictor = IrisPredictor()
    incomplete_df = pd.DataFrame([[5.0, 3.4]], columns=["sepal_length", "sepal_width"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        predictor.predict_batch(incomplete_df)
