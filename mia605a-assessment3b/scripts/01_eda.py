"""
01_eda.py — Exploratory analysis and figures for dual-accelerometer leak features.
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

sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)

NUM_COLS = [
    "Pressure_kPa",
    "Flow_Lps",
    "BackgroundNoise_dB",
    "A1_RMS_ms2",
    "A1_Kurtosis",
    "A1_Peak_ms2",
    "A1_SpectralEnergy",
    "A1_DominantFreq_Hz",
    "A2_RMS_ms2",
    "A2_Kurtosis",
    "A2_Peak_ms2",
    "A2_SpectralEnergy",
    "A1A2_CrossCorrelation",
    "A1A2_EnergyRatio",
]


def run_eda(df: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = df[NUM_COLS + ["Leak_Class"]].describe().round(4).to_dict()
    class_counts = df["Leak_Class"].value_counts().sort_index().to_dict()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Leak_Label", order=df["Leak_Label"].value_counts().index, ax=ax, palette="Set2")
    ax.set_title("Class distribution (multiclass leak labels)")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["A1_RMS_ms2"], kde=True, ax=ax, color="#2E86AB", bins=40)
    ax.set_title("Sensor A1 RMS acceleration (m/s²)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_a1_rms_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Leak_Label", y="A1_RMS_ms2", ax=ax, palette="muted")
    ax.set_title("A1 RMS by leak type")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_a1_rms_by_class.png", dpi=150)
    plt.close(fig)

    corr = df[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(min(800, len(df)), random_state=42),
        x="A1_RMS_ms2",
        y="A2_RMS_ms2",
        hue="Leak_Label",
        alpha=0.65,
        ax=ax,
        palette="tab10",
    )
    ax.set_title("Dual-sensor RMS scatter (A1 vs A2)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_dual_rms_scatter.png", dpi=150)
    plt.close(fig)

    by_class = df.groupby("Leak_Class")[["A1_DominantFreq_Hz", "A2_DominantFreq_Hz"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    by_class.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Mean dominant frequency by leak class")
    ax.set_xlabel("Leak class")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_dominant_freq_by_class.png", dpi=150)
    plt.close(fig)

    pearson_leak = float(df["A1_RMS_ms2"].corr(df["Leak_Class"]))
    spearman_noise = float(stats.spearmanr(df["BackgroundNoise_dB"], df["A1_RMS_ms2"]).statistic)

    return {
        "row_count": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "summary_stats": summary,
        "pearson_a1_rms_vs_class": pearson_leak,
        "spearman_noise_vs_a1_rms": spearman_noise,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "Assessment3B_data.csv")
    parser.add_argument("--fig-dir", type=Path, default=root / "outputs" / "figures")
    parser.add_argument("--summary", type=Path, default=root / "outputs" / "eda_summary.json")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    results = run_eda(df, args.fig_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2))
    print(json.dumps({"eda": "ok", "figures": str(args.fig_dir)}, indent=2))


if __name__ == "__main__":
    main()
