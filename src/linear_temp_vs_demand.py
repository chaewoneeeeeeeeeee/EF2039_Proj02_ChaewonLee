# File: linear_temp_vs_demand.py
# Purpose: Draw a simple 1D linear regression between outdoor temperature (T2M_toc)
#          and national electricity demand (nat_demand) to illustrate a physical
#          BEMS-relevant relationship.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main():
    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "continuous_dataset.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 2) Select x = outdoor temperature, y = electricity demand
    #    Subsample for better visibility in scatter plot
    x = df["T2M_toc"].values.reshape(-1, 1)
    y = df["nat_demand"].values
    step = 10  # sampling every 10 points
    x_s = x[::step]
    y_s = y[::step]

    # 3) Fit linear regression
    model = LinearRegression()
    model.fit(x_s, y_s)
    y_pred = model.predict(x_s)
    r2 = r2_score(y_s, y_pred)

    # 4) Sort for a clean regression line
    order = np.argsort(x_s[:, 0])
    x_line = x_s[order]
    y_line = y_pred[order]

    # 5) Plot scatter + regression line
    plt.figure(figsize=(8, 5))
    plt.scatter(x_s, y_s, s=12, alpha=0.35, label="Data points")
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear fit")

    plt.xlabel("Outdoor Temperature (T2M_toc)")
    plt.ylabel("Electricity Demand (MW)")
    plt.title(f"T2M_toc vs Electricity Demand (Linear Fit)\nR² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()

    # 6) Save figure
    out_path = results_dir / "linear_T2M_vs_demand.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved linear regression plot to: {out_path}")
    print(f"[INFO] R² score = {r2:.4f}")


if __name__ == "__main__":
    main()
