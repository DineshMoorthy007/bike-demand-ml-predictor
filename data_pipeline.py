from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk with basic validation."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc

    if data.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return data


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure all required columns are present before transformations."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_bike_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess bike sharing hourly data and return model-ready features and target.

    Steps:
    1. Drop the 'instant' column.
    2. Extract day and month from 'dteday'.
    3. One-hot encode categorical columns: 'season', 'weathersit'.
    4. Scale numerical columns: 'temp', 'hum', 'windspeed'.
    5. Return X (features) and y (target='cnt') as DataFrames.
    """
    required_columns = [
        "instant",
        "dteday",
        "season",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
        "cnt",
    ]
    validate_required_columns(df, required_columns)

    processed = df.copy()

    # Drop row identifier not useful for modeling.
    processed = processed.drop(columns=["instant"])

    # Parse date and derive calendar features.
    processed["dteday"] = pd.to_datetime(processed["dteday"], errors="coerce")
    if processed["dteday"].isna().any():
        raise ValueError("Column 'dteday' contains invalid date values.")

    processed["day"] = processed["dteday"].dt.day
    processed["month"] = processed["dteday"].dt.month
    processed = processed.drop(columns=["dteday"])

    # One-hot encode specified categorical variables.
    processed = pd.get_dummies(
        processed,
        columns=["season", "weathersit"],
        prefix=["season", "weathersit"],
        drop_first=False,
        dtype=int,
    )

    # Scale selected numerical features for stable model training.
    numeric_cols = ["temp", "hum", "windspeed"]
    scaler = StandardScaler()
    processed[numeric_cols] = scaler.fit_transform(processed[numeric_cols])

    # Keep target as DataFrame to satisfy the required return type.
    y = processed[["cnt"]].copy()
    X = processed.drop(columns=["cnt"]).copy()

    return X, y


def build_features_and_target(file_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess bike sharing data from a CSV file path."""
    data = load_data(file_path)
    return preprocess_bike_data(data)


if __name__ == "__main__":
    dataset_path = Path("data/hour.csv")

    try:
        X_df, y_df = build_features_and_target(dataset_path)
        print(f"Preprocessing completed successfully. X shape: {X_df.shape}, y shape: {y_df.shape}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Pipeline error: {exc}")
