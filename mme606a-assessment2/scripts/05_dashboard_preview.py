"""
05_dashboard_preview.py — Matplotlib-based dashboard preview (report / placeholder for Power BI).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", context="notebook")


def build_dashboard(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # KPI strip
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.set_facecolor("#16213e")
    kpi_ax.axis("off")
    avg_eff = df["Efficiency (%)"].mean()
    total_downtime = df["Downtime (Hours)"].sum()
    fail_rate = df["Predicted_Failure_Binary"].mean() * 100
    avg_energy = df["Energy_Intensity_kWh_per_hr"].mean()
    kpi_text = (
        f"  AVG EFFICIENCY: {avg_eff:.1f}%   |   "
        f"TOTAL DOWNTIME: {total_downtime:,.0f} h   |   "
        f"FAILURE RISK: {fail_rate:.1f}%   |   "
        f"ENERGY INTENSITY: {avg_energy:.2f} kWh/hr"
    )
    kpi_ax.text(0.5, 0.5, kpi_text, ha="center", va="center", fontsize=14, color="white", fontweight="bold")

    # Chart 1: downtime by type
    ax1 = fig.add_subplot(gs[1, 0])
    by_type = df.groupby("Machine_Type")["Downtime (Hours)"].mean().sort_values(ascending=True)
    by_type.plot(kind="barh", ax=ax1, color="#e94560")
    ax1.set_title("Avg Downtime by Machine Type", color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#0f3460")

    # Chart 2: failure by location
    ax2 = fig.add_subplot(gs[1, 1])
    loc_fail = df.groupby("Location")["Predicted_Failure_Binary"].mean()
    loc_fail.plot(kind="bar", ax=ax2, color=["#533483", "#e94560", "#0f3460"])
    ax2.set_title("Failure Rate by Plant", color="white")
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#0f3460")

    # Chart 3: efficiency trend
    ax3 = fig.add_subplot(gs[1, 2])
    trend = df.groupby("Record_Month")["Efficiency (%)"].mean()
    ax3.plot(range(len(trend)), trend.values, color="#00d9ff", linewidth=2)
    ax3.set_title("Efficiency Trend (Monthly)", color="white")
    ax3.tick_params(colors="white", labelbottom=False)
    ax3.set_facecolor("#0f3460")

    # Chart 4: scatter vibration vs faults
    ax4 = fig.add_subplot(gs[2, 0])
    sample = df.sample(min(1500, len(df)), random_state=1)
    ax4.scatter(sample["Vibration_Level (mm/s)"], sample["Faults_Reported"], alpha=0.35, c="#e94560", s=12)
    ax4.set_xlabel("Vibration (mm/s)", color="white")
    ax4.set_ylabel("Faults", color="white")
    ax4.set_title("Vibration vs Fault Count", color="white")
    ax4.tick_params(colors="white")
    ax4.set_facecolor("#0f3460")

    # Chart 5: energy box by location
    ax5 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=df, x="Location", y="Power_Consumption (kWh)", ax=ax5, palette="muted")
    ax5.set_title("Power Consumption by Plant", color="white")
    ax5.tick_params(colors="white")
    ax5.set_facecolor("#0f3460")

    # Chart 6: top machines table text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    top_risk = (
        df.groupby("Machine_ID")
        .agg(
            fail_rate=("Predicted_Failure_Binary", "mean"),
            avg_vib=("Vibration_Level (mm/s)", "mean"),
            avg_down=("Downtime (Hours)", "mean"),
        )
        .sort_values("fail_rate", ascending=False)
        .head(6)
    )
    lines = ["TOP RISK MACHINES (ID | fail% | vib | down)"]
    for mid, row in top_risk.iterrows():
        lines.append(
            f"M{mid:03d}  {row.fail_rate*100:5.1f}%  {row.avg_vib:5.2f}  {row.avg_down:5.1f}h"
        )
    ax6.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10, color="#00d9ff")

    fig.suptitle(
        "MME606A — Manufacturing Performance Dashboard (Preview)",
        fontsize=16,
        color="white",
        fontweight="bold",
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard preview -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=base / "data" / "manufacturing_operations_featured.csv")
    parser.add_argument("--output", type=Path, default=base / "outputs" / "figures" / "dashboard_preview.png")
    parser.add_argument("--media-copy", type=Path, default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if "Predicted_Failure_Binary" not in df.columns:
        df["Predicted_Failure_Binary"] = (df["Predicted_Failure"] == "Yes").astype(int)
    build_dashboard(df, args.output)
    if args.media_copy:
        import shutil
        shutil.copy(args.output, args.media_copy)


if __name__ == "__main__":
    main()
