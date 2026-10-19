"""
04_train_models.py — Classification (failure) and regression (downtime) models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
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
    "Machine_Type_Code",
    "Location_Code",
    "Energy_Intensity_kWh_per_hr",
    "Maintenance_Cost_per_hr",
    "Availability_Ratio",
    "Fault_Rate_per_100h",
    "Days_Since_Maintenance",
    "Vibration_Temp_Index",
]


def train_classification(df: pd.DataFrame, random_state: int = 42) -> dict:
    X = df[FEATURE_COLS]
    y = df["Predicted_Failure_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    models = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    results = {}
    best_model = None
    best_auc = -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "roc_auc": float(auc),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True),
        }
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model)

    assert best_model is not None
    joblib.dump(best_model[1], Path(__file__).resolve().parents[1] / "outputs" / "best_classifier.joblib")
    results["best_classifier"] = best_model[0]
    return results


REGRESSION_FEATURES = [
    c for c in FEATURE_COLS if c not in ("Downtime (Hours)", "Faults_Reported", "Maintenance_Cost (USD)")
]


def train_regression(df: pd.DataFrame, random_state: int = 42) -> dict:
    X = df[REGRESSION_FEATURES]
    y = df["Downtime (Hours)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=2.0))]),
        "random_forest_reg": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=3, random_state=random_state
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=base / "data" / "manufacturing_operations_featured.csv")
    parser.add_argument("--report", type=Path, default=base / "outputs" / "model_metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metrics = {
        "classification": train_classification(df),
        "regression_downtime": train_regression(df),
        "split": "75/25 stratified (classification), random (regression)",
        "features": FEATURE_COLS,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["classification"], indent=2))


if __name__ == "__main__":
    main()
