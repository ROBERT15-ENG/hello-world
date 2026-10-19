"""
01_generate_data.py — Synthetic manufacturing operations dataset (MME606A A2 Paper A).

Generates >=5,000 monthly machine records across multiple plants with realistic
engineering relationships, intentional noise, missing values, and outliers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

MACHINE_TYPES = ["CNC", "Lathe", "Milling", "3D Printer", "Press", "Robot Arm"]
LOCATIONS = ["Plant_A_Melbourne", "Plant_B_Geelong", "Plant_C_Ballarat"]
N_MACHINES = 85
START_DATE = datetime(2022, 1, 1)
N_MONTHS = 62  # 85 * 62 = 5270 rows


def _machine_base_params(machine_type: str) -> dict:
    """Typical operating envelopes by machine family."""
    profiles = {
        "CNC": {"hours_mu": 420, "hours_sd": 45, "power_per_hr": 12.5, "temp_mu": 48},
        "Lathe": {"hours_mu": 380, "hours_sd": 50, "power_per_hr": 9.0, "temp_mu": 42},
        "Milling": {"hours_mu": 400, "hours_sd": 48, "power_per_hr": 11.0, "temp_mu": 45},
        "3D Printer": {"hours_mu": 520, "hours_sd": 60, "power_per_hr": 4.5, "temp_mu": 55},
        "Press": {"hours_mu": 350, "hours_sd": 40, "power_per_hr": 18.0, "temp_mu": 38},
        "Robot Arm": {"hours_mu": 480, "hours_sd": 55, "power_per_hr": 6.5, "temp_mu": 40},
    }
    return profiles[machine_type]


def generate_dataset(n_machines: int = N_MACHINES, n_months: int = N_MONTHS) -> pd.DataFrame:
    rows: list[dict] = []
    machine_meta = []
    for mid in range(1, n_machines + 1):
        mtype = RNG.choice(MACHINE_TYPES, p=[0.22, 0.18, 0.16, 0.12, 0.16, 0.16])
        loc = LOCATIONS[mid % len(LOCATIONS)]
        age_years = float(RNG.uniform(0.5, 12))
        reliability_factor = max(0.55, 1.0 - age_years * 0.035 + RNG.normal(0, 0.05))
        machine_meta.append(
            {"Machine_ID": mid, "Machine_Type": mtype, "Location": loc, "Age_Years": age_years,
             "Reliability_Factor": reliability_factor}
        )

    for month_idx in range(n_months):
        period_start = START_DATE + pd.DateOffset(months=month_idx)
        seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * month_idx / 12)
        for meta in machine_meta:
            prof = _machine_base_params(meta["Machine_Type"])
            op_hours = max(
                80,
                RNG.normal(prof["hours_mu"] * seasonal * meta["Reliability_Factor"], prof["hours_sd"]),
            )
            op_hours = float(np.clip(op_hours, 60, 720))

            base_power = op_hours * prof["power_per_hr"] * RNG.lognormal(0, 0.12)
            power = float(base_power * (1.0 + 0.15 * (1 - meta["Reliability_Factor"])))

            temp = float(
                prof["temp_mu"]
                + 0.02 * op_hours
                + RNG.normal(0, 3.5)
                + 8 * (1 - meta["Reliability_Factor"])
            )
            vibration = float(
                2.5
                + 0.008 * op_hours
                + 0.12 * (temp - prof["temp_mu"])
                + RNG.gamma(2.0, 0.8)
                + 3.0 * (1 - meta["Reliability_Factor"])
            )

            output_units = max(10, op_hours * RNG.uniform(0.8, 1.4) * (meta["Machine_Type"] != "3D Printer"))
            if meta["Machine_Type"] == "3D Printer":
                output_units = max(5, op_hours * RNG.uniform(0.15, 0.35))

            efficiency = float(
                np.clip(
                    72
                    + 8 * (output_units / (op_hours + 1))
                    - 0.04 * vibration
                    - 0.03 * (temp - 40)
                    + RNG.normal(0, 4),
                    35,
                    99,
                )
            )

            fault_lambda = 0.3 + 2.5 * (1 - meta["Reliability_Factor"]) + 0.01 * vibration
            faults = int(RNG.poisson(fault_lambda))
            downtime = float(
                max(0, faults * RNG.uniform(1.5, 6.0) + RNG.exponential(2.0) * (1 - meta["Reliability_Factor"]))
            )
            downtime = min(downtime, op_hours * 0.45)

            maint_cost = float(
                200
                + faults * RNG.uniform(180, 650)
                + downtime * RNG.uniform(25, 90)
                + meta["Age_Years"] * RNG.uniform(40, 120)
            )

            days_since_maint = int(RNG.integers(5, 120))
            last_maint = period_start - timedelta(days=int(days_since_maint))

            failure_score = (
                0.30 * (vibration / 12)
                + 0.22 * min(faults / 3, 1.5)
                + 0.18 * min(downtime / 25, 1.5)
                + 0.15 * ((100 - efficiency) / 25)
                + 0.10 * min(days_since_maint / 75, 1.2)
                - 0.20 * meta["Reliability_Factor"]
                + RNG.normal(0, 0.12)
            )
            # Target ~10–14% positive class for realistic rare-event maintenance planning
            predicted_failure = "Yes" if failure_score > 0.52 else "No"

            rows.append(
                {
                    "Record_Month": period_start.strftime("%Y-%m"),
                    "Machine_ID": meta["Machine_ID"],
                    "Machine_Type": meta["Machine_Type"],
                    "Location": meta["Location"],
                    "Operating_Hours": round(op_hours, 2),
                    "Power_Consumption (kWh)": round(power, 2),
                    "Temperature (°C)": round(temp, 2),
                    "Vibration_Level (mm/s)": round(vibration, 3),
                    "Maintenance_Cost (USD)": round(maint_cost, 2),
                    "Downtime (Hours)": round(downtime, 2),
                    "Faults_Reported": faults,
                    "Last_Maintenance_Date": last_maint.date().isoformat(),
                    "Efficiency (%)": round(efficiency, 2),
                    "Output_Units": int(output_units),
                    "Age_Years": round(meta["Age_Years"], 2),
                    "Predicted_Failure": predicted_failure,
                }
            )

    df = pd.DataFrame(rows)
    df = _inject_imperfections(df)
    return df


def _inject_imperfections(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing values, outliers, and sensor noise per assessment brief."""
    n = len(df)
    miss_cols = [
        "Temperature (°C)",
        "Vibration_Level (mm/s)",
        "Maintenance_Cost (USD)",
        "Efficiency (%)",
    ]
    for col in miss_cols:
        idx = RNG.choice(n, size=int(n * RNG.uniform(0.012, 0.025)), replace=False)
        df.loc[idx, col] = np.nan

    # Erroneous duplicate readings (same machine/month) — small fraction
    dup_idx = RNG.choice(n, size=12, replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # Outliers: spike in power / vibration
    out_idx = RNG.choice(len(df), size=45, replace=False)
    df.loc[out_idx[:20], "Power_Consumption (kWh)"] *= RNG.uniform(1.8, 2.6, size=20)
    df.loc[out_idx[20:35], "Vibration_Level (mm/s)"] *= RNG.uniform(2.0, 3.5, size=15)
    df.loc[out_idx[35:], "Downtime (Hours)"] = RNG.uniform(80, 160, size=10)

    # Occasional negative downtime typo (data entry error)
    typo_idx = RNG.choice(len(df), size=8, replace=False)
    df.loc[typo_idx, "Downtime (Hours)"] = -RNG.uniform(0.5, 4.0, size=8)

    # Rounding noise on operating hours
    df["Operating_Hours"] = df["Operating_Hours"] + RNG.normal(0, 0.3, len(df))

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manufacturing_operations.csv",
    )
    args = parser.parse_args()
    df = generate_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")
    print(f"Failure rate (Yes): {(df['Predicted_Failure'] == 'Yes').mean():.2%}")
    print(f"Missing cells: {df.isna().sum().sum()}")


if __name__ == "__main__":
    main()
