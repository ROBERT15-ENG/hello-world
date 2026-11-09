"""
00_generate_data.py — Create Assessment3B_data.csv when the Moodle file is unavailable.

The official course dataset was not present in this workspace. This script materialises
a dual-accelerometer, windowed feature table consistent with LeakTestbed-style labelling:
  0 = no leak, 1 = orifice, 2 = longitudinal crack, 3 = circumferential crack, 4 = gasket.
Replace output with the provided CSV when available (see README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {
    0: "No_Leak",
    1: "Orifice_Leak",
    2: "Longitudinal_Crack",
    3: "Circumferential_Crack",
    4: "Gasket_Leak",
}

FEATURE_SPECS = {
    0: {"rms": 0.08, "kurt": 3.1, "peak": 0.25, "spec_e": 0.04, "dom_f": 42.0, "cc": 0.92},
    1: {"rms": 0.55, "kurt": 6.8, "peak": 1.85, "spec_e": 0.38, "dom_f": 185.0, "cc": 0.71},
    2: {"rms": 0.48, "kurt": 5.9, "peak": 1.62, "spec_e": 0.31, "dom_f": 96.0, "cc": 0.74},
    3: {"rms": 0.51, "kurt": 6.2, "peak": 1.70, "spec_e": 0.34, "dom_f": 112.0, "cc": 0.72},
    4: {"rms": 0.44, "kurt": 5.4, "peak": 1.45, "spec_e": 0.28, "dom_f": 78.0, "cc": 0.76},
}


def _sensor_block(rng: np.random.Generator, base: dict, scale: float) -> dict:
    noise = rng.normal(0, 1, size=6)
    rms = max(0.01, base["rms"] * scale + 0.04 * noise[0])
    kurt = max(2.5, base["kurt"] + 0.35 * noise[1])
    peak = max(0.05, base["peak"] * scale + 0.12 * noise[2])
    spec = max(0.001, base["spec_e"] * scale + 0.03 * noise[3])
    dom_f = max(20.0, base["dom_f"] + 8.0 * noise[4])
    zcr = max(0.05, 0.18 + 0.04 * abs(noise[5]) + (0.08 if base["rms"] > 0.3 else 0.0))
    return {"rms": rms, "kurt": kurt, "peak": peak, "spec_e": spec, "dom_f": dom_f, "zcr": zcr}


def generate(n_per_class: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []
    record_id = 1
    for leak_class, name in CLASS_NAMES.items():
        spec = FEATURE_SPECS[leak_class]
        for window in range(n_per_class):
            pressure = rng.uniform(210, 420)
            flow = rng.uniform(0.0, 0.55)
            noise_db = rng.uniform(28, 48)
            a1_scale = rng.uniform(0.85, 1.15)
            a2_scale = rng.uniform(0.82, 1.12)
            a1 = _sensor_block(rng, spec, a1_scale)
            a2 = _sensor_block(rng, spec, a2_scale)
            cross_corr = float(np.clip(spec["cc"] + rng.normal(0, 0.06), 0.35, 0.99))
            energy_ratio = float(np.clip(a1["rms"] / max(a2["rms"], 1e-6) + rng.normal(0, 0.08), 0.4, 2.2))
            rows.append(
                {
                    "Record_ID": record_id,
                    "Window_Index": window + 1,
                    "Pressure_kPa": round(pressure, 2),
                    "Flow_Lps": round(flow, 4),
                    "BackgroundNoise_dB": round(noise_db, 2),
                    "A1_RMS_ms2": round(a1["rms"], 5),
                    "A1_Kurtosis": round(a1["kurt"], 4),
                    "A1_Peak_ms2": round(a1["peak"], 5),
                    "A1_SpectralEnergy": round(a1["spec_e"], 5),
                    "A1_DominantFreq_Hz": round(a1["dom_f"], 2),
                    "A1_ZeroCrossRate": round(a1["zcr"], 5),
                    "A2_RMS_ms2": round(a2["rms"], 5),
                    "A2_Kurtosis": round(a2["kurt"], 4),
                    "A2_Peak_ms2": round(a2["peak"], 5),
                    "A2_SpectralEnergy": round(a2["spec_e"], 5),
                    "A2_DominantFreq_Hz": round(a2["dom_f"], 2),
                    "A2_ZeroCrossRate": round(a2["zcr"], 5),
                    "A1A2_CrossCorrelation": round(cross_corr, 4),
                    "A1A2_EnergyRatio": round(energy_ratio, 4),
                    "Leak_Class": leak_class,
                    "Leak_Label": name,
                }
            )
            record_id += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "Assessment3B_data.csv")
    parser.add_argument("--n-per-class", type=int, default=400)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"Data already exists: {args.output}")
        return
    df = generate(args.n_per_class)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
