# MIA605A Assessment 3 — Paper B (Pipeline Leakage ML)

Dual-accelerometer pipeline for **multiclass leak-type** (classes 0–4) and **binary leak detection** on the official Moodle CSV.

**Student placeholders:** `STUDENT_NAME`, `STUDENT_ID`

## Data

Primary source: `data/Assessment3B_data.csv` (Moodle upload, ~100k raw samples: `Sample`, `A1`, `A2`, `Leak status`). See `data/DATA_SOURCE.md`.

`scripts/00_generate_data.py` remains a **synthetic fallback only** for offline development — do not use for submission.

## Pipeline

Run as **scripts** or open matching notebooks in `notebooks/` (Colab / Jupyter).

```bash
cd mia605a-assessment3b
python3 -m pip install -r requirements.txt
python3 scripts/01_eda.py
python3 scripts/02_preprocess.py
python3 scripts/03_train_models.py
```

| Step | Script | Notebook | Output |
|------|--------|----------|--------|
| EDA | `01_eda.py` | `01_eda.ipynb` | `outputs/figures/fig01–07_*.png`, `outputs/eda_summary.json` |
| Preprocess | `02_preprocess.py` | `02_preprocess.ipynb` | `data/Assessment3B_clean.csv`, `outputs/preprocess_report.json` |
| ML | `03_train_models.py` | `03_train_models.ipynb` | `outputs/model_metrics.json`, confusion-matrix figures |
| All (optional) | — | `04_train_models.ipynb` | Runs scripts 01–03 via subprocess |

Sliding windows (1024 samples, hop 256) extract fourteen accelerometer features before train/val/test splits.

## Class labels

| Code | Meaning |
|------|---------|
| 0 | No leak |
| 1 | Orifice leak |
| 2 | Longitudinal crack |
| 3 | Circumferential crack |
| 4 | Gasket leak |

## Report

Academic write-up: `docs/MIA605A_Assessment3B_Report.md` (real Moodle metrics, November 2026).
