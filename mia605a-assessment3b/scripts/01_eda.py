"""
01_eda.py — Exploratory analysis for official dual-accelerometer leak recordings.
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

from importlib.util import module_from_spec, spec_from_file_location

sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)

ROOT = Path(__file__).resolve().parents[1]
_preprocess_path = ROOT / "scripts" / "02_preprocess.py"
_spec = spec_from_file_location("preprocess_mod", _preprocess_path)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

NUM_COLS = _mod.FEATURE_COLS
load_raw_csv = _mod.load_raw_csv
build_windowed_table = _mod.build_windowed_table
CLASS_NAMES = _mod.CLASS_NAMES


def run_eda(raw: pd.DataFrame, featured: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = featured[NUM_COLS + ["Leak_Class"]].describe().round(4).to_dict()
    class_counts = featured["Leak_Class"].value_counts().sort_index().to_dict()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=featured, x="Leak_Label", order=featured["Leak_Label"].value_counts().index, ax=ax, palette="Set2")
    ax.set_title("Window-level class distribution (official Moodle data)")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(featured["A1_RMS_ms2"], kde=True, ax=ax, color="#2E86AB", bins=40)
    ax.set_title("Sensor A1 RMS acceleration (sliding windows)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_a1_rms_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=featured, x="Leak_Label", y="A1_RMS_ms2", ax=ax, palette="muted")
    ax.set_title("A1 RMS by leak type")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_a1_rms_by_class.png", dpi=150)
    plt.close(fig)

    corr = featured[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sample_df = featured.sample(min(800, len(featured)), random_state=42)
    sns.scatterplot(
        data=sample_df,
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

    by_class = featured.groupby("Leak_Class")[["A1_DominantFreq_Hz", "A2_DominantFreq_Hz"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    by_class.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Mean dominant frequency by leak class")
    ax.set_xlabel("Leak class")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_dominant_freq_by_class.png", dpi=150)
    plt.close(fig)

  # Raw time-domain snippet (first ~0.05 s per recording)
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    seg_starts = [0]
    for i in range(1, len(raw)):
        if raw.iloc[i]["Sample"] < raw.iloc[i - 1]["Sample"]:
            seg_starts.append(i)
    seg_starts.append(len(raw))
    for ax, (s, e) in zip(axes, zip(seg_starts[:-1], seg_starts[1:])):
        block = raw.iloc[s:e]
        leak = int(block["Leak status"].mode().iloc[0])
        t = block["Sample"].to_numpy()
        mask = t <= (t[0] + 0.05)
        ax.plot(t[mask], block["A1"].to_numpy()[mask], lw=0.8, label="A1", color="#2E86AB")
        ax.plot(t[mask], block["A2"].to_numpy()[mask], lw=0.8, alpha=0.85, label="A2", color="#E94F37")
        ax.set_title(f"Raw trace — {CLASS_NAMES.get(leak, leak)}")
        ax.set_ylabel("Acceleration")
        ax.legend(loc="upper right", fontsize="small")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_raw_trace_snippets.png", dpi=150)
    plt.close(fig)

    pearson_leak = float(featured["A1_RMS_ms2"].corr(featured["Leak_Class"]))
    raw_rows = int(len(raw))
    leak_counts_raw = raw["Leak status"].value_counts().sort_index().to_dict()

    return {
        "row_count_raw": raw_rows,
        "row_count_windows": int(len(featured)),
        "raw_class_counts": {str(k): int(v) for k, v in leak_counts_raw.items()},
        "window_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "summary_stats": summary,
        "pearson_a1_rms_vs_class": pearson_leak,
        "window_size": _mod.WINDOW_SIZE,
        "hop_size": _mod.HOP_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Assessment3B_data.csv")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "eda_summary.json")
    args = parser.parse_args()

    raw = load_raw_csv(args.data)
    featured, _ = build_windowed_table(raw)
    results = run_eda(raw, featured, args.fig_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2))
    print(json.dumps({"eda": "ok", "figures": str(args.fig_dir), "windows": results["row_count_windows"]}, indent=2))


if __name__ == "__main__":
    main()
