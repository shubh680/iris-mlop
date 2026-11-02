"""Utility functions for data loading, preprocessing, and feature definitions."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define feature names and target name
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_width"]
TARGET_NAME = "petal_length"

# Descriptive labels with units for UI display
FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_width": "Petal Width (cm)",
    "petal_length": "Petal Length (cm)",
}


def load_iris_regression_data() -> pd.DataFrame:
    """Load the Iris dataset and reformat it for linear regression.

    Target: Petal Length (cm)
    Features: Sepal Length (cm), Sepal Width (cm), Petal Width (cm)
    Also includes the original 'species' label for visualization exploration.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Rename columns to snake_case
    rename_dict = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=rename_dict)

    # Map target integer to species name
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    if "target" in df.columns:
        df["species"] = df["target"].map(species_map)
        df = df.drop(columns=["target"])

    return df


def get_train_test_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split Iris dataset into train and test sets for regression."""
    df = load_iris_regression_data()
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
