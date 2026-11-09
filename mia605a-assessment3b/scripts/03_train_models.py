"""
03_train_models.py — Multiclass leak-type and binary leak/no-leak classifiers (≥2 families each).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]
_preprocess_path = ROOT / "scripts" / "02_preprocess.py"
_spec = spec_from_file_location("preprocess_mod", _preprocess_path)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
FEATURE_COLS = _mod.FEATURE_COLS


def _plot_confusion(y_true, y_pred, labels, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_multiclass(X_train, y_train, X_val, y_val, X_test, y_test, fig_dir: Path, random_state: int) -> dict:
    models = {
        "logistic_multiclass": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_multiclass": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: dict = {}
    best_name = None
    best_f1 = -1.0
    best_estimator = None
  # Tuned SVM (second family with hyperparameter search)
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
        ]
    )
    grid = GridSearchCV(
        svm,
        {"clf__C": [0.5, 2.0, 8.0], "clf__gamma": ["scale", 0.05, 0.01]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    models["svm_rbf_tuned"] = grid.best_estimator_
    results["svm_rbf_tuned"] = {"best_params": grid.best_params_, "cv_f1_macro_mean": float(grid.best_score_)}

    labels = sorted(np.unique(y_test))
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        f1_val = f1_score(y_val, pred_val, average="macro")
        f1_test = f1_score(y_test, pred_test, average="macro")
        acc_test = accuracy_score(y_test, pred_test)
        results[name] = {
            **results.get(name, {}),
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "val_accuracy": float(accuracy_score(y_val, pred_val)),
            "val_f1_macro": float(f1_val),
            "test_accuracy": float(acc_test),
            "test_f1_macro": float(f1_test),
            "classification_report": classification_report(y_test, pred_test, output_dict=True, zero_division=0),
        }
        _plot_confusion(
            y_test,
            pred_test,
            labels,
            f"Multiclass confusion — {name}",
            fig_dir / f"cm_multiclass_{name}.png",
        )
        if f1_test > best_f1:
            best_f1 = f1_test
            best_name = name
            best_estimator = model

    return {"models": results, "best_model": best_name, "best_estimator": best_estimator}


def train_binary(X_train, y_train, X_val, y_val, X_test, y_test, fig_dir: Path, random_state: int) -> dict:
    models = {
        "logistic_binary": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest_binary": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: dict = {}
    best_name = None
    best_f1 = -1.0
    best_estimator = None

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        proba_test = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "val_accuracy": float(accuracy_score(y_val, pred_val)),
            "val_f1": float(f1_score(y_val, pred_val)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "test_f1": float(f1_score(y_test, pred_test)),
            "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
            "classification_report": classification_report(y_test, pred_test, output_dict=True, zero_division=0),
        }
        _plot_confusion(y_test, pred_test, [0, 1], f"Binary confusion — {name}", fig_dir / f"cm_binary_{name}.png")
        if results[name]["test_f1"] > best_f1:
            best_f1 = results[name]["test_f1"]
            best_name = name
            best_estimator = model

    return {"models": results, "best_model": best_name, "best_estimator": best_estimator}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Assessment3B_clean.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs" / "preprocess_report.json")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--metrics", type=Path, default=ROOT / "outputs" / "model_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    split = json.loads(args.split.read_text())
    tr, va, te = split["split_indices"]["train"], split["split_indices"]["val"], split["split_indices"]["test"]

    X = df[FEATURE_COLS]
    y_multi = df["Leak_Class"]
    y_bin = df["Binary_Leak"]

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    multi = train_multiclass(X.iloc[tr], y_multi.iloc[tr], X.iloc[va], y_multi.iloc[va], X.iloc[te], y_multi.iloc[te], args.fig_dir, args.random_state)
    binary = train_binary(X.iloc[tr], y_bin.iloc[tr], X.iloc[va], y_bin.iloc[va], X.iloc[te], y_bin.iloc[te], args.fig_dir, args.random_state)

    best_multi_est = multi.pop("best_estimator", None)
    best_bin_est = binary.pop("best_estimator", None)
    payload = {
        "multiclass": multi,
        "binary": binary,
        "recommended": {
            "multiclass_model": multi["best_model"],
            "binary_model": binary["best_model"],
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2))
    if best_multi_est is not None:
        joblib.dump(best_multi_est, ROOT / "outputs" / "best_multiclass.joblib")
    if best_bin_est is not None:
        joblib.dump(best_bin_est, ROOT / "outputs" / "best_binary.joblib")
    print(json.dumps(payload["recommended"], indent=2))


if __name__ == "__main__":
    main()
