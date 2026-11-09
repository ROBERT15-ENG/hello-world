# MIA605A Assessment 3 — Paper B (Pipeline Leakage ML)

Dual-accelerometer feature dataset for **multiclass leak-type** (classes 0–4) and **binary leak detection** tasks.

**Student placeholders:** `STUDENT_NAME`, `STUDENT_ID`

## Data

Place the official Moodle file at `data/Assessment3B_data.csv`. If missing, run:

```bash
python3 scripts/00_generate_data.py
```

This workspace used a **synthetic fallback** because the provided CSV was not found locally — replace with the official file and re-run the pipeline before submission.

## Pipeline

```bash
cd mia605a-assessment3b
python3 -m pip install -r requirements.txt
python3 scripts/00_generate_data.py
python3 scripts/01_eda.py
python3 scripts/02_preprocess.py
python3 scripts/03_train_models.py
```

| Step | Script | Output |
|------|--------|--------|
| Data | `00_generate_data.py` | `data/Assessment3B_data.csv` |
| EDA | `01_eda.py` | `outputs/figures/fig01–06_*.png`, `outputs/eda_summary.json` |
| Preprocess | `02_preprocess.py` | `data/Assessment3B_clean.csv`, `outputs/preprocess_report.json` |
| ML | `03_train_models.py` | `outputs/model_metrics.json`, confusion-matrix figures |

## Class labels

| Code | Meaning |
|------|---------|
| 0 | No leak |
| 1 | Orifice leak |
| 2 | Longitudinal crack |
| 3 | Circumferential crack |
| 4 | Gasket leak |

## Report

Academic write-up: Project store `docs/MIA605A_Assessment3B_Report.md` (mirrored under repo `docs/` when synced).
