"""
02_preprocess.py — Cleaning, validation, and export of analysis-ready dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

NUMERIC_COLS = [
    "Operating_Hours",
    "Power_Consumption (kWh)",
    "Temperature (°C)",
    "Vibration_Level (mm/s)",
    "Maintenance_Cost (USD)",
    "Downtime (Hours)",
    "Faults_Reported",
    "Efficiency (%)",
    "Output_Units",
    "Age_Years",
]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"initial_rows": len(df), "steps": []}
    df = df.copy()

    # Drop exact duplicates on machine + month
    key = ["Machine_ID", "Record_Month"]
    n_dup = df.duplicated(subset=key, keep="first").sum()
    df = df.drop_duplicates(subset=key, keep="first")
    report["steps"].append({"action": "drop_duplicate_machine_month", "removed": int(n_dup)})

    # Fix invalid downtime
    neg = (df["Downtime (Hours)"] < 0).sum()
    df.loc[df["Downtime (Hours)"] < 0, "Downtime (Hours)"] = np.nan
    report["steps"].append({"action": "nullify_negative_downtime", "count": int(neg)})

    # Cap extreme outliers using IQR (engineering log retained in report)
    for col in ["Power_Consumption (kWh)", "Vibration_Level (mm/s)", "Downtime (Hours)"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        capped = (df[col] > upper).sum()
        df.loc[df[col] > upper, col] = upper
        report["steps"].append({"action": f"cap_outliers_{col}", "capped": int(capped), "upper": float(upper)})

    # Parse dates
    df["Last_Maintenance_Date"] = pd.to_datetime(df["Last_Maintenance_Date"], errors="coerce")
    df["Record_Month_DT"] = pd.to_datetime(df["Record_Month"] + "-01")

    # Median imputation for numeric sensor/economic fields (robust to skew)
    imp = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = imp.fit_transform(df[NUMERIC_COLS])
    report["steps"].append({"action": "median_imputation", "columns": NUMERIC_COLS})

    # Encode categoricals for ML export
    le_type = LabelEncoder()
    le_loc = LabelEncoder()
    df["Machine_Type_Code"] = le_type.fit_transform(df["Machine_Type"].astype(str))
    df["Location_Code"] = le_loc.fit_transform(df["Location"].astype(str))
    df["Predicted_Failure_Binary"] = (df["Predicted_Failure"] == "Yes").astype(int)

    report["final_rows"] = len(df)
    report["missing_after"] = int(df[NUMERIC_COLS].isna().sum().sum())
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=base / "data" / "manufacturing_operations.csv")
    parser.add_argument("--output", type=Path, default=base / "data" / "manufacturing_operations_clean.csv")
    parser.add_argument("--report", type=Path, default=base / "outputs" / "preprocess_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    clean, report = preprocess(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Clean dataset: {len(clean):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
