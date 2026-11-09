"""
02_preprocess.py — Sliding-window feature extraction, imputation, and stratified splits.

Input: official Moodle CSV with columns Sample, A1, A2, Leak status (raw accelerometer traces).
Output: windowed feature table (Assessment3B_clean.csv) plus preprocess_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    0: "No_Leak",
    1: "Orifice_Leak",
    2: "Longitudinal_Crack",
    3: "Circumferential_Crack",
    4: "Gasket_Leak",
}

WINDOW_SIZE = 1024
HOP_SIZE = 256
MIN_WINDOW_SIZE = WINDOW_SIZE

FEATURE_COLS = [
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


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        c = col.strip()
        if c.lower() == "sample":
            rename[col] = "Sample"
        elif c == "A1":
            rename[col] = "A1"
        elif c == "A2":
            rename[col] = "A2"
        elif "leak" in c.lower():
            rename[col] = "Leak status"
    df = df.rename(columns=rename)
    required = ["Sample", "A1", "A2", "Leak status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; got {list(df.columns)}")
    return df


def infer_sample_rate_hz(samples: np.ndarray) -> float:
    dts = np.diff(samples.astype(float))
    dts = dts[dts > 0]
    if len(dts) == 0:
        return 25_600.0
    dt = float(np.median(dts))
    return 1.0 / dt if dt > 0 else 25_600.0


def segment_recordings(df: pd.DataFrame) -> list[dict]:
    """Split contiguous recordings where Sample time resets (five leak-condition captures)."""
    starts = [0]
    for i in range(1, len(df)):
        if df.iloc[i]["Sample"] < df.iloc[i - 1]["Sample"]:
            starts.append(i)
    starts.append(len(df))
    segments: list[dict] = []
    for rec_id, (s, e) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        block = df.iloc[s:e]
        leak = int(block["Leak status"].mode().iloc[0])
        segments.append(
            {
                "record_id": rec_id,
                "start_idx": s,
                "end_idx": e,
                "leak_class": leak,
                "sample": block["Sample"].to_numpy(dtype=float),
                "a1": block["A1"].to_numpy(dtype=float),
                "a2": block["A2"].to_numpy(dtype=float),
            }
        )
    return segments


def _sensor_features(arr: np.ndarray, fs: float) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(arr**2)))
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False))
    peak = float(np.max(np.abs(arr)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(arr))) > 0))
    fft = np.fft.rfft(arr - np.mean(arr))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    spec_energy = float(np.sum(power) / len(arr))
    if len(power) > 1:
        dom_idx = int(np.argmax(power[1:]) + 1)
        dom_f = float(freqs[dom_idx])
    else:
        dom_f = 0.0
    return {
        "rms": rms,
        "kurt": kurt,
        "peak": peak,
        "spec_e": spec_energy,
        "dom_f": dom_f,
        "zcr": zcr,
    }


def window_features(a1: np.ndarray, a2: np.ndarray, fs: float) -> dict[str, float]:
    f1 = _sensor_features(a1, fs)
    f2 = _sensor_features(a2, fs)
    if np.std(a1) > 1e-12 and np.std(a2) > 1e-12:
        cc = float(np.corrcoef(a1, a2)[0, 1])
    else:
        cc = 0.0
    e1 = float(np.sum(a1**2))
    e2 = float(np.sum(a2**2))
    return {
        "A1_RMS_ms2": f1["rms"],
        "A1_Kurtosis": f1["kurt"],
        "A1_Peak_ms2": f1["peak"],
        "A1_SpectralEnergy": f1["spec_e"],
        "A1_DominantFreq_Hz": f1["dom_f"],
        "A1_ZeroCrossRate": f1["zcr"],
        "A2_RMS_ms2": f2["rms"],
        "A2_Kurtosis": f2["kurt"],
        "A2_Peak_ms2": f2["peak"],
        "A2_SpectralEnergy": f2["spec_e"],
        "A2_DominantFreq_Hz": f2["dom_f"],
        "A2_ZeroCrossRate": f2["zcr"],
        "A1A2_CrossCorrelation": cc,
        "A1A2_EnergyRatio": e1 / (e2 + 1e-12),
    }


def build_windowed_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    segments = segment_recordings(df)
    rows: list[dict] = []
    meta: dict = {"segments": [], "window_size": WINDOW_SIZE, "hop_size": HOP_SIZE}

    for seg in segments:
        fs = infer_sample_rate_hz(seg["sample"])
        a1, a2 = seg["a1"], seg["a2"]
        n = len(a1)
        window_idx = 0
        seg_windows = 0
        for start in range(0, n - WINDOW_SIZE + 1, HOP_SIZE):
            end = start + WINDOW_SIZE
            feats = window_features(a1[start:end], a2[start:end], fs)
            rows.append(
                {
                    "Record_ID": seg["record_id"],
                    "Window_Index": window_idx,
                    "Sample_Start": float(seg["sample"][start]),
                    "Sample_Rate_Hz": fs,
                    **feats,
                    "Leak_Class": seg["leak_class"],
                    "Leak_Label": CLASS_NAMES.get(seg["leak_class"], str(seg["leak_class"])),
                }
            )
            window_idx += 1
            seg_windows += 1
        meta["segments"].append(
            {
                "record_id": seg["record_id"],
                "leak_class": seg["leak_class"],
                "raw_samples": n,
                "sample_rate_hz": fs,
                "windows": seg_windows,
            }
        )

    featured = pd.DataFrame(rows)
    meta["window_count"] = int(len(featured))
    return featured, meta


def preprocess(
    df: pd.DataFrame, random_state: int = 42, chronological_split: bool = True
) -> tuple[pd.DataFrame, dict]:
    if "Leak_Class" not in df.columns:
        featured, window_meta = build_windowed_table(df)
    else:
        featured = df.copy()
        window_meta = {"window_count": len(featured), "source": "pre_featured"}

    clean = featured.copy()
    clean[FEATURE_COLS] = clean[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    medians = clean[FEATURE_COLS].median()
    clean[FEATURE_COLS] = clean[FEATURE_COLS].fillna(medians)
    clean["Binary_Leak"] = (clean["Leak_Class"] > 0).astype(int)

    if chronological_split:
        tr_idx: list[int] = []
        va_idx: list[int] = []
        te_idx: list[int] = []
        offset = 0
        for rec_id, grp in clean.groupby("Record_ID", sort=True):
            n = len(grp)
            n_test = max(1, int(round(0.20 * n)))
            n_val = max(1, int(round(0.15 * (n - n_test))))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = max(1, n - n_test - n_val)
            local = np.arange(offset, offset + n)
            tr_idx.extend(local[:n_train].tolist())
            va_idx.extend(local[n_train : n_train + n_val].tolist())
            te_idx.extend(local[n_train + n_val :].tolist())
            offset += n
        tr_idx, va_idx, te_idx = sorted(tr_idx), sorted(va_idx), sorted(te_idx)
    else:
        idx = np.arange(len(clean))
        y_multi = clean["Leak_Class"]
        te_idx = []
        tr_idx, te_idx_list = train_test_split(idx, test_size=0.20, random_state=random_state, stratify=y_multi)
        te_idx = te_idx_list.tolist()
        y_tr = clean.iloc[tr_idx]["Leak_Class"]
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=random_state, stratify=y_tr)
        tr_idx, va_idx = sorted(tr_idx), sorted(va_idx)
        te_idx = sorted(te_idx)

    report = {
        "rows": int(len(clean)),
        "feature_columns": FEATURE_COLS,
        "imputed_values": {k: float(v) for k, v in medians.to_dict().items()},
        "binary_leak_rate": float(clean["Binary_Leak"].mean()),
        "split_sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "split_indices": {"train": tr_idx, "val": va_idx, "test": te_idx},
        "split_strategy": "chronological_per_recording" if chronological_split else "stratified_random",
        "windowing": window_meta,
    }
    return clean, report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "Assessment3B_data.csv")
    parser.add_argument("--out-csv", type=Path, default=root / "data" / "Assessment3B_clean.csv")
    parser.add_argument("--report", type=Path, default=root / "outputs" / "preprocess_report.json")
    args = parser.parse_args()

    raw = load_raw_csv(args.data)
    clean, report = preprocess(raw)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out_csv, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["split_sizes"], indent=2))


if __name__ == "__main__":
    main()
