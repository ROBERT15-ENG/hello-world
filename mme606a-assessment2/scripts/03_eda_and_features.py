"""
03_eda_and_features.py — EDA statistics, engineered features, and visualisations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Energy_Intensity_kWh_per_hr"] = df["Power_Consumption (kWh)"] / np.maximum(df["Operating_Hours"], 1)
    df["Maintenance_Cost_per_hr"] = df["Maintenance_Cost (USD)"] / np.maximum(df["Operating_Hours"], 1)
    total_time = df["Operating_Hours"] + df["Downtime (Hours)"]
    df["Availability_Ratio"] = df["Operating_Hours"] / np.maximum(total_time, 1)
    df["Fault_Rate_per_100h"] = 100 * df["Faults_Reported"] / np.maximum(df["Operating_Hours"], 1)
    df["Days_Since_Maintenance"] = (
        df["Record_Month_DT"] - df["Last_Maintenance_Date"]
    ).dt.days.clip(lower=0)
    df["Vibration_Temp_Index"] = df["Vibration_Level (mm/s)"] * (df["Temperature (°C)"] / 100)
    return df


def summary_statistics(df: pd.DataFrame) -> dict:
    cols = [
        "Operating_Hours",
        "Power_Consumption (kWh)",
        "Downtime (Hours)",
        "Efficiency (%)",
        "Vibration_Level (mm/s)",
    ]
    summary = df[cols].agg(["mean", "median", "std", "min", "max"]).round(3)
    return summary.to_dict()


def run_eda(df: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {"summary": summary_statistics(df)}

    # Distribution: Efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Efficiency (%)"], kde=True, ax=ax, color="#2E86AB", bins=35)
    ax.set_title("Distribution of Machine Efficiency (%)")
    ax.set_xlabel("Efficiency (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_efficiency_distribution.png", dpi=150)
    plt.close(fig)
    results["efficiency_skew"] = float(stats.skew(df["Efficiency (%)"]))

    # Distribution: Vibration
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Vibration_Level (mm/s)"], kde=True, ax=ax, color="#A23B72", bins=35)
    ax.set_title("Distribution of Vibration Level (mm/s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_vibration_distribution.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    num = df[
        [
            "Operating_Hours",
            "Power_Consumption (kWh)",
            "Temperature (°C)",
            "Vibration_Level (mm/s)",
            "Downtime (Hours)",
            "Efficiency (%)",
            "Energy_Intensity_kWh_per_hr",
            "Availability_Ratio",
        ]
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix — Key Operational Variables")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Downtime by machine type
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("Machine_Type")["Downtime (Hours)"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Machine_Type", y="Downtime (Hours)", order=order, ax=ax, palette="Set2")
    ax.set_title("Monthly Downtime by Machine Type")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_downtime_by_machine_type.png", dpi=150)
    plt.close(fig)

    # Failure rate by location
    fail_rate = df.groupby("Location")["Predicted_Failure_Binary"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    fail_rate.plot(kind="bar", ax=ax, color=["#F18F01", "#C73E1D", "#3B1F2B"])
    ax.set_title("Failure Indicator Rate by Plant Location")
    ax.set_ylabel("Proportion Predicted_Failure = Yes")
    ax.set_ylim(0, fail_rate.max() * 1.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_failure_rate_by_location.png", dpi=150)
    plt.close(fig)

    # Energy vs operating hours scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(min(2000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="Operating_Hours",
        y="Power_Consumption (kWh)",
        hue="Machine_Type",
        alpha=0.5,
        ax=ax,
        legend=False,
    )
    ax.set_title("Energy Consumption vs Operating Hours")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_energy_vs_hours.png", dpi=150)
    plt.close(fig)

    # Monthly trend
    monthly = df.groupby("Record_Month")["Downtime (Hours)"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax, color="#2E86AB")
    ax.set_title("Average Plant Downtime Trend (Monthly)")
    ax.set_ylabel("Mean Downtime (Hours)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_downtime_trend.png", dpi=150)
    plt.close(fig)

    results["failure_rate_overall"] = float(df["Predicted_Failure_Binary"].mean())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=base / "data" / "manufacturing_operations_clean.csv")
    parser.add_argument("--output", type=Path, default=base / "data" / "manufacturing_operations_featured.csv")
    parser.add_argument("--fig-dir", type=Path, default=base / "outputs" / "figures")
    parser.add_argument("--report", type=Path, default=base / "outputs" / "eda_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Last_Maintenance_Date", "Record_Month_DT"])
    featured = engineer_features(df)
    featured.to_csv(args.output, index=False)
    eda_results = run_eda(featured, args.fig_dir)
    eda_results["engineered_features"] = [
        "Energy_Intensity_kWh_per_hr",
        "Maintenance_Cost_per_hr",
        "Availability_Ratio",
        "Fault_Rate_per_100h",
        "Days_Since_Maintenance",
        "Vibration_Temp_Index",
    ]
    args.report.write_text(json.dumps(eda_results, indent=2))
    print(f"Featured dataset -> {args.output}")
    print(f"Figures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
