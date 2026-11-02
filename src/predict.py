"""Inference pipeline for predicting petal length using trained Linear Regression model."""

from pathlib import Path
from typing import Union, List
import joblib
import pandas as pd

from src.utils import FEATURE_NAMES
from src.train import train_model, MODEL_PATH


class IrisPredictor:
    """Predictor wrapper for Iris Petal Length Linear Regression."""

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_or_train()

    def _load_or_train(self):
        """Load trained model or train one if not present."""
        if not self.model_path.exists():
            print(f"Model artifact not found at {self.model_path}. Training now...")
            train_model(model_dir=self.model_path.parent)

        return joblib.load(self.model_path)

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_width: float,
    ) -> float:
        """Predict petal length for single observation in centimeters."""
        self._validate_inputs(sepal_length, sepal_width, petal_width)
        df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_width]],
            columns=FEATURE_NAMES,
        )
        prediction = self.model.predict(df)[0]
        # Petal length physically cannot be negative
        return max(0.0, float(prediction))

    def predict_batch(self, features_df: pd.DataFrame) -> List[float]:
        """Predict petal lengths for a batch dataframe."""
        missing = [f for f in FEATURE_NAMES if f not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        preds = self.model.predict(features_df[FEATURE_NAMES])
        return [max(0.0, float(p)) for p in preds]

    def _validate_inputs(
        self, sepal_length: float, sepal_width: float, petal_width: float
    ) -> None:
        """Validate input values to ensure realistic floral dimensions."""
        for name, val in [
            ("sepal_length", sepal_length),
            ("sepal_width", sepal_width),
            ("petal_width", petal_width),
        ]:
            if val is None or not isinstance(val, (int, float)):
                raise ValueError(f"Value for {name} must be numeric.")
            if val <= 0:
                raise ValueError(f"Value for {name} must be greater than 0.")


# Convenience function
def predict_petal_length(
    sepal_length: float,
    sepal_width: float,
    petal_width: float,
    model_path: Union[str, Path] = MODEL_PATH,
) -> float:
    """Convenience helper to predict petal length."""
    predictor = IrisPredictor(model_path=model_path)
    return predictor.predict(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_width=petal_width,
    )
