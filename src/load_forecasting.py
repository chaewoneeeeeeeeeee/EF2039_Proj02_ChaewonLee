# File: load_forecasting.py
# Term Project 02 - Hourly Electricity Load Forecasting for BEMS
#
# This script loads the continuous_dataset.csv file, constructs
# time-series features, trains several regression models, and
# evaluates their performance on an hourly load forecasting task.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Added small comment to clarify forecasting pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Added note: future improvement - try different model architectures



def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load raw CSV file and parse datetime column."""
    print("[STEP 1] Load dataset")
    df = pd.read_csv(data_path)
    print(f"[INFO] Raw shape: {df.shape}")

    # Parse datetime and sort by time
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Drop rows with missing target values, if any
    df = df.dropna(subset=["nat_demand"])

    print(f"[INFO] After datetime parsing and target cleaning: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct time-based and demand-based features for load forecasting.

    Features:
      - hour, dayofweek, month, is_weekend
      - weather variables near Toc (T2M_toc, QV2M_toc, TQL_toc, W2M_toc)
      - lagged demand (lag1, lag24)
      - rolling mean demand (roll_3, roll_24)
      - holiday, school (binary indicators)
    """
    print("\n[STEP 2] Build features")

    df = df.copy()

    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek  # 0 = Monday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Lag features for nat_demand (in MW)
    df["lag1"] = df["nat_demand"].shift(1)
    df["lag24"] = df["nat_demand"].shift(24)

    # Rolling mean features to smooth noise
    df["roll_3"] = df["nat_demand"].rolling(window=3).mean()
    df["roll_24"] = df["nat_demand"].rolling(window=24).mean()

    # Remove first rows where lag/rolling values are not available
    df = df.dropna().reset_index(drop=True)

    print(f"[INFO] After feature engineering: {df.shape}")
    return df


def prepare_xy(df: pd.DataFrame):
    """Prepare design matrix X and target vector y."""
    print("\n[STEP 3] Prepare X and y")

    feature_cols = [
        # Time features
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        # Weather features (Toc site)
        "T2M_toc",
        "QV2M_toc",
        "TQL_toc",
        "W2M_toc",
        # Demand history features
        "lag1",
        "lag24",
        "roll_3",
        "roll_24",
        # Calendar indicators
        "holiday",
        "school",
    ]

    x = df[feature_cols].values.astype("float32")
    y = df["nat_demand"].values.astype("float32")

    print(f"[INFO] X shape: {x.shape}")
    print(f"[INFO] y shape: {y.shape}")
    return x, y


def split_train_test(x, y, test_ratio: float = 0.2):
    """
    Split into train and test sets.
    We keep the temporal order (shuffle=False) for time-series forecasting.
    """
    print("\n[STEP 4] Split train and test sets")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio, shuffle=False
    )

    print(f"[INFO] Train size: {x_train.shape[0]}  Test size: {x_test.shape[0]}")
    return x_train, x_test, y_train, y_test


def train_models(x_train, y_train):
    """Train three regression models and return the fitted models."""
    print("\n[STEP 5] Train models")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=60,
            max_depth=18,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        print(f"[INFO] Trained {name}")

    return models


def evaluate_models(models, x_train, y_train, x_test, y_test):
    """
    Evaluate all models on both train and test sets.

    Returns:
        results: dict mapping model name to performance dictionary
    """
    print("\n[STEP 6] Evaluate models")

    results = {}

    for name, model in models.items():
        # Train performance
        pred_train = model.predict(x_train)
        mae_tr = mean_absolute_error(y_train, pred_train)

        # Old sklearn version does not support 'squared' argument,
        # so we compute RMSE as sqrt(MSE) manually.
        mse_tr = mean_squared_error(y_train, pred_train)
        rmse_tr = np.sqrt(mse_tr)
        r2_tr = r2_score(y_train, pred_train)

        # Test performance
        pred_test = model.predict(x_test)
        mae_te = mean_absolute_error(y_test, pred_test)
        mse_te = mean_squared_error(y_test, pred_test)
        rmse_te = np.sqrt(mse_te)
        r2_te = r2_score(y_test, pred_test)

        results[name] = {
            "Train MAE": mae_tr,
            "Train RMSE": rmse_tr,
            "Train R2": r2_tr,
            "Test MAE": mae_te,
            "Test RMSE": rmse_te,
            "Test R2": r2_te,
        }

        # Pretty print similar to professor's example
        print(f"\n===== Model: {name} =====")
        print(f"Train MAE  : {mae_tr:.4f}")
        print(f"Train RMSE : {rmse_tr:.4f}")
        print(f"Train R^2  : {r2_tr:.4f}")
        print(f"Test  MAE  : {mae_te:.4f}")
        print(f"Test  RMSE : {rmse_te:.4f}")
        print(f"Test  R^2  : {r2_te:.4f}")

    return results



def select_best_model(results, models):
    """Select the best model according to Test R2 score."""
    print("\n[STEP 7] Select best model")

    best_name = None
    best_r2 = -np.inf

    for name, metrics in results.items():
        r2_test = metrics["Test R2"]
        if r2_test > best_r2:
            best_r2 = r2_test
            best_name = name

    best_model = models[best_name]
    print(f"[INFO] Best model: {best_name} (Test R^2 = {best_r2:.4f})")
    return best_name, best_model


def plot_predictions(
    model, df: pd.DataFrame, x_test, y_test, results_dir: Path
) -> None:
    """Plot actual vs predicted load on the test set and save the figure."""
    print("\n[STEP 8] Plot predictions")

    # Use the test part of the datetime index for plotting
    total_len = df.shape[0]
    test_len = x_test.shape[0]
    time_test = df["datetime"].iloc[total_len - test_len :]

    y_pred = model.predict(x_test)

    plt.figure(figsize=(12, 5))
    plt.plot(time_test, y_test, label="Actual load", linewidth=1.0)
    plt.plot(time_test, y_pred, label="Predicted load", linewidth=1.0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Electricity load (MW)")
    plt.title("Actual vs Predicted Hourly Load (Test Set)")
    plt.legend()
    plt.tight_layout()

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "hourly_load_prediction.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved prediction plot to: {plot_path}")


def main():
    """Main entry point for the load forecasting experiment."""

    # Resolve project paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "continuous_dataset.csv"
    results_dir = project_root / "results"

    # 1) Load raw dataset
    df_raw = load_dataset(data_path)

    # 2) Feature engineering
    df_feat = build_features(df_raw)

    # 3) Prepare X and y
    x, y = prepare_xy(df_feat)

    # 4) Split into train and test
    x_train, x_test, y_train, y_test = split_train_test(x, y, test_ratio=0.2)

    # 5) Train models
    models = train_models(x_train, y_train)

    # 6) Evaluate models
    results = evaluate_models(models, x_train, y_train, x_test, y_test)

    # 7) Select the best model based on Test R2
    best_name, best_model = select_best_model(results, models)

    # 8) Plot predictions for the best model
    plot_predictions(best_model, df_feat, x_test, y_test, results_dir)

    print("\n[DONE] Load forecasting pipeline finished successfully.")


if __name__ == "__main__":
    main()
