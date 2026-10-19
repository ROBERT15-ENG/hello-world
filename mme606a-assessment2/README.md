# MME606A Assessment 2 — Paper A (Manufacturing Analytics)

Synthetic manufacturing operations dataset, reproducible Python pipeline, and Power BI build artefacts for **STUDENT_NAME** (`STUDENT_ID`).

## Quick start

```bash
cd /workspace/mme606a-assessment2
python3 -m pip install -r requirements.txt
python3 scripts/01_generate_data.py
python3 scripts/02_preprocess.py
python3 scripts/03_eda_and_features.py
python3 scripts/04_train_models.py
python3 scripts/05_dashboard_preview.py --media-copy /cursor/stores/self/media/dashboard_preview.png
```

## Pipeline

Run as scripts **or** open the matching notebooks in `notebooks/` (Google Colab / Jupyter).

| Step | Script | Notebook | Purpose |
|------|--------|----------|---------|
| 1 | `01_generate_data.py` | `01_generate_data.ipynb` | Create `data/manufacturing_operations.csv` (≥5,000 rows) |
| 2 | `02_preprocess.py` | `02_preprocess.ipynb` | Clean, impute, encode → `manufacturing_operations_clean.csv` |
| 3 | `03_eda_and_features.py` | `03_eda_and_features.ipynb` | Engineered features, EDA figures → `outputs/figures/` |
| 4 | `04_train_models.py` | `04_train_models.ipynb` | Failure classification + downtime regression metrics |
| 5 | `05_dashboard_preview.py` | `05_dashboard_preview.ipynb` | Dashboard-style PNG for report appendix |

## Outputs

- `outputs/figures/` — EDA and dashboard preview figures
- `outputs/model_metrics.json` — ML evaluation results
- `powerbi/` — DAX measures and layout specification (build `.pbix` in Power BI Desktop)

## Submission placeholders

- Replace `STUDENT_NAME` and `STUDENT_ID` in the report cover section.
- Add Loom link: `[INSERT LOOM VIDEO LINK]`
- Upload `.pbix` via OneDrive per assessment instructions.
