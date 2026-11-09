# Assessment 3B data source

**Primary dataset:** Official Moodle CSV uploaded for MIA605A Assessment 3 — Paper B.

| Item | Detail |
|------|--------|
| File | `Assessment3B_data.csv` |
| Rows | ~100,000 time samples |
| Columns | `Sample`, `A1`, `A2`, `Leak status` |
| Labels | `0` = no leak; `1`–`4` = leak types |
| Structure | Five continuous recordings (one per class); `Sample` resets between recordings |

The pipeline (`scripts/01_eda.py` → `02_preprocess.py` → `03_train_models.py`) extracts **sliding-window** time- and frequency-domain features from the raw accelerometer traces. Do **not** use `scripts/00_generate_data.py` as the primary data source; that script remains only as an optional synthetic fallback for offline development.

**Provenance:** Course-provided file `Assessment3B_data_b5d8fbf8da5a555e2d9094dd4dc83605_2626.csv` (Moodle), copied to this folder on 2026-11-09.
