# File: seasonal_temp_vs_demand.py
# Description: Plot temperature vs electricity demand by season with linear regression lines.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset and parse datetime column."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def assign_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add a season column based on the month."""
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["season"] = df["datetime"].dt.month.apply(get_season)
    return df

def plot_seasonal_graphs(df: pd.DataFrame, results_dir: Path):
    """Create 4 seasonal scatter plots with linear regression lines."""
    seasons = ["Winter", "Spring", "Summer", "Fall"]

    plt.figure(figsize=(12, 10))

    for i, season in enumerate(seasons, 1):
        sub = df[df["season"] == season]

        X = sub[["T2M_toc"]].values
        y = sub["nat_demand"].values

        # Fit linear regression line
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        plt.subplot(2, 2, i)
        plt.scatter(sub["T2M_toc"], sub["nat_demand"], s=5, alpha=0.5)
        plt.plot(sub["T2M_toc"], y_pred, color="red", linewidth=1.5)
        plt.title(f"{season} (R² = {model.score(X, y):.3f})")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Electricity Demand (MW)")

    plt.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "seasonal_T2M_vs_demand.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved seasonal plots to: {save_path}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "continuous_dataset.csv"
    results_dir = project_root / "results"

    df = load_dataset(data_path)
    df = assign_season(df)

    plot_seasonal_graphs(df, results_dir)

if __name__ == "__main__":
    main()
