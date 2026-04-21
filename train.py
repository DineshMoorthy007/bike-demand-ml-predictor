from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from data_pipeline import build_features_and_target


def load_training_data(dataset_path: Path) -> tuple[Any, Any]:
    """Load preprocessed features and target from the shared pipeline."""
    return build_features_and_target(dataset_path)


def train_with_tuning(X: Any, y: Any) -> tuple[XGBRegressor, float, float]:
    """Split the data, tune an XGBoost regressor, and evaluate on test data."""
    # 80/20 split with fixed random state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.values.ravel(),
        test_size=0.2,
        random_state=42,
    )

    # Define base model and a compact hyperparameter grid for tuning.
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [200, 400],
    }

    # GridSearchCV finds the best parameter combination by cross-validation.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)
    best_model: XGBRegressor = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return best_model, rmse, r2


def save_model(model: XGBRegressor, output_dir: Path, filename: str) -> Path:
    """Create output directory if needed and persist trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / filename
    joblib.dump(model, model_path)
    return model_path


def main() -> None:
    dataset_path = Path("data/hour.csv")
    model_dir = Path("models")
    model_name = "xgb_bike_model.joblib"

    try:
        X, y = load_training_data(dataset_path)
        best_model, rmse, r2 = train_with_tuning(X, y)
        saved_path = save_model(best_model, model_dir, model_name)

        print(f"Training complete. RMSE: {rmse:.4f}")
        print(f"Training complete. R-squared: {r2:.4f}")
        print(f"Model saved to: {saved_path}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data or preprocessing error: {exc}")
    except Exception as exc:
        print(f"Unexpected training error: {exc}")


if __name__ == "__main__":
    main()
