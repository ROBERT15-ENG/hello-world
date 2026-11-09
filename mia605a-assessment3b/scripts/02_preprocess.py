"""
02_preprocess.py — Imputation, scaling metadata, and train/val/test index export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "Pressure_kPa",
    "Flow_Lps",
    "BackgroundNoise_dB",
    "A1_RMS_ms2",
    "A1_Kurtosis",
    "A1_Peak_ms2",
    "A1_SpectralEnergy",
    "A1_DominantFreq_Hz",
    "A1_ZeroCrossRate",
    "A2_RMS_ms2",
    "A2_Kurtosis",
    "A2_Peak_ms2",
    "A2_SpectralEnergy",
    "A2_DominantFreq_Hz",
    "A2_ZeroCrossRate",
    "A1A2_CrossCorrelation",
    "A1A2_EnergyRatio",
]


def preprocess(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    clean = df.copy()
    clean[FEATURE_COLS] = clean[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    medians = clean[FEATURE_COLS].median()
    clean[FEATURE_COLS] = clean[FEATURE_COLS].fillna(medians)
    clean["Binary_Leak"] = (clean["Leak_Class"] > 0).astype(int)

    # Stratified hold-out indices for reproducible reporting
    idx = np.arange(len(clean))
    y_multi = clean["Leak_Class"]
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=random_state, stratify=y_multi)
    y_tr = clean.iloc[tr_idx]["Leak_Class"]
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=random_state, stratify=y_tr)

    report = {
        "rows": int(len(clean)),
        "feature_columns": FEATURE_COLS,
        "imputed_values": {k: float(v) for k, v in medians.to_dict().items()},
        "binary_leak_rate": float(clean["Binary_Leak"].mean()),
        "split_sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "split_indices": {
            "train": tr_idx.tolist(),
            "val": va_idx.tolist(),
            "test": te_idx.tolist(),
        },
    }
    return clean, report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "Assessment3B_data.csv")
    parser.add_argument("--out-csv", type=Path, default=root / "data" / "Assessment3B_clean.csv")
    parser.add_argument("--report", type=Path, default=root / "outputs" / "preprocess_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    clean, report = preprocess(df)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out_csv, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["split_sizes"], indent=2))


if __name__ == "__main__":
    main()
