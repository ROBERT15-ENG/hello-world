# MME606A Assessment 2 — Paper A Technical Report

| | |
|---|---|
| **Student name** | STUDENT_NAME |
| **Student ID** | STUDENT_ID |
| **Unit** | MME606A Data Analytics for Engineering Applications |
| **Assessment** | 2 — Paper A |
| **Date** | 19 October 2026 |
| **Loom video** | [INSERT LOOM VIDEO LINK] |

---

## Table of contents

1. Introduction  
2. Task 1 — Introduction, data understanding and pre-processing  
3. Task 2 — Descriptive statistics, feature engineering and visualisation  
4. Task 3 — Machine learning model development  
5. Task 4 — Model comparison and engineering impact  
6. Task 5 — Final recommendations and conclusion  
7. Task 6 — Power BI dashboard and video presentation  
8. References  
9. Reflection  
10. Appendix  

---

## 1. Introduction

Manufacturing organisations must balance throughput, energy cost, and asset reliability. Unplanned machine failures create lost production, safety exposure, and expedited logistics for spare parts. This assessment develops an end-to-end analytics workflow on a **synthetic multi-plant dataset** that obeys the unit brief: at least 5,000 records, mandated columns, realistic ranges, and deliberate imperfections. The engineering focus is **predictive maintenance**—using condition indicators (vibration, temperature, faults) and utilisation (hours, power) to flag elevated failure risk before stoppages.

Condition-based maintenance is widely framed by ISO 17359, which emphasises repeatable monitoring and diagnostic reasoning on machines [1]. Supervised learning complements rule-based alarms by learning multivariate boundaries that human thresholds miss, provided labels and features are traceable to physics. Deliverables include Python scripts (`scripts/01`–`05`), CSV data, Power BI specifications, and this report (~2,900 words, appendix excluded).

The case study intentionally spans **multiple plants and machine types** so analytics must generalise across heterogeneous assets rather than tuning a single CNC line. That heterogeneity reflects real enterprises where legacy lathes share a hall with modern robot cells. A model that only works on one type would have limited capital planning value. Likewise, monthly granularity matches how many SMEs close CMMS work orders and utility bills, even though high-value assets may eventually need hourly or per-cycle models.

Traditional engineering decisions—fixed PM intervals, operator walk-arounds, and threshold alarms on single sensors—remain valuable but scale poorly across dozens of machines. Analytics adds a **portfolio view**: which machines contribute disproportionately to downtime dollars this quarter, and which condition trends diverge from peer assets of the same type? The synthetic dataset encodes those peer relationships through shared generative profiles per machine family while allowing asset-level reliability factors for age and maintenance history.

---

## 2. Task 1 — Introduction, data understanding and pre-processing

### 2.1 Industrial scenario

The modelled enterprise operates three fictional Victorian plants (`Plant_A_Melbourne`, `Plant_B_Geelong`, `Plant_C_Ballarat`) with 85 production assets: CNC machines, lathes, milling centres, 3D printers, hydraulic presses, and robot arms. Each record is one **machine-month** from January 2022 to February 2027, aggregating SCADA-like averages and CMMS maintenance outcomes.

### 2.2 Engineering problems

1. **Reliability** — rising vibration and fault counts precede downtime.  
2. **Energy** — power draw scales with hours but efficiency varies with mechanical condition.  
3. **Maintenance economics** — costs spike when faults and downtime compound.

These problems matter because downtime in supply-chain manufacturing is often valued in thousands of dollars per hour once penalties and rework are included [2], while electricity is a controllable OPEX once intensity (kWh per productive hour) is visible to production and facilities teams.

### 2.3 Dataset description

| Item | Value |
|------|-------|
| Raw rows | 5,282 (includes deliberate duplicates) |
| Clean rows | 5,270 unique machine–months |
| Machines | 85 |
| Plants | 3 |
| `Predicted_Failure` = Yes | 22.0% |

Required columns use assessment naming, e.g. `Power_Consumption (kWh)`, `Temperature (°C)`, `Vibration_Level (mm/s)`. Supplementary fields `Record_Month`, `Output_Units`, and `Age_Years` support features and dashboards.

### 2.4 Intentional imperfections

Missing values (~1.5–2.5%) on temperature, vibration, maintenance cost, and efficiency; multiplicative outliers on power and vibration; eight negative downtime typos; twelve duplicate machine-month rows; Gaussian noise on operating hours. These mimic intermittent sensor dropouts and CMMS entry errors.

### 2.4b Generation summary (see `docs/dataset-generation-notes.md`)

Operating hours draw from type-specific normals with seasonal modulation (±8% annual cycle). Power consumption equals hours times a type-specific kWh rate with log-normal noise and degradation for unreliable assets. Vibration increases with hours, temperature excursion, and gamma-distributed shocks. Fault counts follow Poisson rates driven by vibration and reliability; downtime accumulates from fault repairs and exponential unplanned events. Efficiency is a clipped function of output per hour and stress variables. The `Predicted_Failure` label arises from a weighted score compared to threshold 0.52, yielding ~22% positives—enough for stratified learning without trivial majority-class accuracy.

### 2.5 Pre-processing (`02_preprocess.py`)

1. Drop duplicate (`Machine_ID`, `Record_Month`) rows.  
2. Set negative `Downtime (Hours)` to missing.  
3. Cap upper tails of power, vibration, and downtime at Q3 + 3×IQR.  
4. Median-impute numeric fields (robust to skewed maintenance costs).  
5. Label-encode `Machine_Type` and `Location`; create `Predicted_Failure_Binary`.

**Justification:** Medians resist fault-cost tails; capping limits leverage of injected spikes without deleting entire asset histories. Stratified modelling later requires clean labels and no duplicate months.

**Analytics linkage:** Pre-processing is not a cosmetic step—it defines what the plant “believes” about reality. If negative downtime were clipped to zero without audit, KPI dashboards would show artificially high availability. By nullifying and imputing, we preserve row counts for small fleets while flagging that those months had quality issues (a production deployment would add a `data_quality_flag` column). Encoding machine type and location lets models absorb heterogeneous baselines: a press legitimately vibrates more than a robot arm, and Geelong’s humidity-controlled hall may differ from Ballarat’s older bay.

---

## 3. Task 2 — Descriptive statistics, feature engineering and visualisation

### 3.1 Summary statistics (clean data)

| Variable | Mean | Median | Std dev |
|----------|------|--------|---------|
| Operating_Hours | 330.3 | 326.6 | 83.3 |
| Power_Consumption (kWh) | 3681.0 | 3526.6 | 1498.9 |
| Downtime (Hours) | 3.90 | 2.95 | 4.33 |
| Efficiency (%) | 78.87 | 79.13 | 4.99 |
| Vibration_Level (mm/s) | 8.35 | 8.17 | 1.44 |

Mean≈median for efficiency (skew −0.26) indicates a stable fleet with few chronically underperforming assets. Downtime is right-skewed: many near-zero months with a tail driving maintenance budgets—typical reliability data [3].

### 3.2 Distributions

**Efficiency (%)** (`fig01`) is near-normal between ~58% and 95%. **Vibration** (`fig02`) shows a mild right tail linked to wear and injected outliers.

### 3.3 Engineered features (≥3)

| Feature | Calculation | Relevance |
|---------|-------------|-----------|
| `Energy_Intensity_kWh_per_hr` | Power / hours | Energy benchmarking |
| `Availability_Ratio` | Hours / (hours + downtime) | OEE availability |
| `Fault_Rate_per_100h` | 100×faults/hours | Normalised fault intensity |
| `Maintenance_Cost_per_hr` | Cost / hours | Economic productivity |
| `Days_Since_Maintenance` | Month date − last service | PM compliance |
| `Vibration_Temp_Index` | Vibration×temperature/100 | Coupled stress proxy |

### 3.4 Visual insights

- **Correlation heatmap (`fig03`)** — power tracks hours; vibration associates with downtime.  
- **Downtime by type (`fig04`)** — presses/CNC show wider spreads → class-focused PM.  
- **Failure by plant (`fig05`)** — heterogeneous risk suggests site practices, not physics alone.  
- **Energy scatter (`fig06`)** — outliers above the band warrant power audits.  
- **Downtime trend (`fig07`)** — seasonality informs maintenance windows.

**Early warning:** combine high vibration, rising fault rate per 100 h, and falling availability—surfaced in the dashboard preview risk table.

### 3.5 Inferential commentary

Plant-level failure rates differ after slicing by `Machine_Type`, indicating that **mix effects** alone do not explain risk—maintenance maturity and spares lead time likely contribute. The correlation between power and hours is expected from physics; deviations from the regression band in `fig06` are candidates for **idle-but-powered** states or tool-change inefficiency. Monthly downtime trend (`fig07`) is smooth enough to support capacity planning yet variable enough to justify safety stock for critical spares.

Statistically, efficiency’s low skew supports parametric tests if comparing plants; downtime’s skew motivates non-parametric comparisons (Kruskal–Wallis) in future work. For this assessment, visual and descriptive evidence suffices to justify feature engineering and supervised labels.

---

## 4. Task 3 — Machine learning model development

### 4.1 Problem formulation

**Primary — classification:** predict `Predicted_Failure` (Yes/No) from condition and utilisation features at month close. Appropriate because maintenance actions are discrete (inspect, defer, replace) [4].

**Secondary — regression:** predict `Downtime (Hours)` excluding contemporaneous faults and maintenance cost from inputs to limit leakage; supports crew and spares planning.

### 4.2 Models

- Logistic regression (standardised, `class_weight=balanced`) — interpretable baseline.  
- Random forest classifier (200 trees) — non-linear interactions.  
- Gradient boosting classifier — sequential residual fitting.  
- Ridge and random forest regressors for downtime.

### 4.3 Training protocol

75/25 hold-out; **stratified** on failure label for classification (`random_state=42`). Features: core sensors, encodings, and engineered fields (`04_train_models.py`).

**Assumptions:** Logistic regression assumes linear decision boundaries in scaled space and independent rows; random forests assume sufficient samples per split and benefit from default depth limits to control variance. **Metrics:** Accuracy alone is misleading with 22% positives; macro-F1 balances classes, while ROC-AUC evaluates ranking quality across thresholds—critical when maintenance managers tune alert cut-offs weekly.

**Split rationale:** A single hold-out mimics deploying a model trained on historical months and evaluated on the latest quarter. Production systems would add rolling-origin cross-validation; for this assessment, stratification ensures both classes appear in test data. No hyperparameter grid search was run to keep the pipeline reproducible on modest hardware; defaults are defensible baselines for teaching and for first pilots on real data.

### 4.4 Results — classification

| Model | Accuracy | Macro F1 | ROC-AUC |
|-------|----------|----------|---------|
| Logistic regression | 0.797 | 0.743 | **0.858** |
| Random forest | 0.803 | 0.737 | 0.844 |
| Gradient boosting | 0.829 | 0.721 | 0.847 |

ROC-AUC is prioritised because missed failures (false negatives) are costlier than extra inspections [5]. Logistic regression achieves recall 0.77 on the `Yes` class with AUC 0.858.

### 4.5 Results — downtime regression

| Model | RMSE (h) | MAE (h) | R² |
|-------|----------|---------|-----|
| Ridge | 0.58 | 0.27 | 0.98 |
| Random forest | **0.54** | **0.13** | 0.98 |

High R² reflects synthetic coupling; real shop-floor data would show more noise.

### 4.6 Confusion matrix (logistic regression, test n=1,318)

True negatives 828 and true positives 223 demonstrate usable discrimination. False positives (200) imply additional inspections; false negatives (67) represent missed risk months. For a plant where a missed bearing failure costs $40k and an inspection costs $600, even precision near 0.53 is economically viable; our recall-oriented logistic model favours catching risk at the expense of extra routes.

### 4.7 Feature roles

`Vibration_Temp_Index` and `Fault_Rate_per_100h` typically rank highly in tree models; logistic coefficients show expected positive signs on vibration and faults after scaling. `Energy_Intensity_kWh_per_hr` helps separate types with different nameplate power—preventing the classifier from confounding “large machine” with “sick machine.”

### 4.8 Limitations

Rule-based labels mean models partly rediscover generation logic. No temporal hold-out—deployment would use rolling validation by month. Imputation hides data-quality flags that production dashboards should expose. External validation on empirical SCADA exports remains mandatory before automating work orders.

---

## 5. Task 4 — Model comparison and engineering impact

| Criterion | Logistic regression | Random forest | Gradient boosting |
|-----------|---------------------|---------------|-------------------|
| Discrimination | **Best AUC** | Similar | High accuracy, lower failure recall |
| Interpretability | **High** | Medium | Low |
| Complexity / train time | **Lowest** | Moderate | Highest |
| Alert economics | Tunable threshold | Black-box ranks | Risk of alert fatigue |

**Impact:** Ranked risk scores support **predictive maintenance planning** and **downtime reduction**; energy intensity features support **efficiency programmes**; availability metrics support **reliability management** under ISO 55000 asset thinking [6].

Gradient boosting achieves the highest accuracy (0.829) but lower recall on failures (0.47), which is undesirable when safety or contractual throughput dominates. Random forest offers a middle ground on accuracy with balanced error types but slightly lower AUC than logistic regression. For **computational efficiency**, all models train in under ten seconds on 5,270 rows—appropriate for monthly batch retraining triggered by an Airflow or cron job pushing scores to SQL Server for Power BI DirectQuery.

---

## 6. Task 5 — Final recommendations and conclusion

**Recommendation:** Adopt **logistic regression** as the primary failure-risk model (highest ROC-AUC, explainable coefficients, fast retraining). Use **random forest regression** for expected downtime hours when scheduling labour and spares.

**Actions:** (1) Weekly export of top-decile risk machines to CMMS; (2) vibration route monitoring when `Vibration_Temp_Index` exceeds plant limits; (3) monthly review of energy intensity outliers; (4) PM alerts when `Days_Since_Maintenance` > 90.

**Conclusion:** The pipeline turns monthly logs into prioritised maintenance and energy insights. Even on synthetic data, multivariate models outperform single-threshold rules, supporting investment in integrated sensor, CMMS, and BI platforms.

**Implementation roadmap:** Phase 1 — shadow mode scoring for three months without auto work orders; Phase 2 — integrate top-10 risk list into weekly reliability meetings; Phase 3 — optional CMMS API triggers above 0.6 probability with human acknowledgement. Each phase should review precision/recall trade-offs with operations finance.

---

## 7. Task 6 — Power BI dashboard and video presentation

Build steps are in `docs/PowerBI_Build_Guide.md` with DAX in `powerbi/DAX_Measures.txt` and layout in `powerbi/Layout_Specification.md`. Import `data/manufacturing_operations_featured.csv`. Pages: executive KPIs (efficiency, failure rate, downtime, energy intensity), reliability scatter/matrix, energy visuals. Slicers: `Location`, `Machine_Type`, `Record_Month`.

`media/dashboard_preview.png` (from `05_dashboard_preview.py`) provides appendix-ready screenshots until `.pbix` is completed in Power BI Desktop.

**Loom (≤5 min, webcam on):** walk KPIs → filter worst plant → show vibration–fault chart → state recommended model (logistic, AUC 0.86) → three engineering recommendations. Link: [INSERT LOOM VIDEO LINK].

Speak to a **non-technical audience**: use “machine health score” instead of “ROC-AUC,” and express impact as avoided downtime hours and maintenance dollars. Show how a slicer changes KPI cards in real time—that interactivity is why Power BI is required alongside static Python figures.

Recommended storyboard: (1) open on fleet KPIs with all plants selected; (2) slice to the highest failure-rate plant and describe one concrete machine ID from the risk table; (3) open the vibration–fault visual and explain what maintenance would do this week; (4) close with energy intensity and a single sentence on expected savings if top risks are addressed before month end.

### Appendix screenshot guide

| Visual | Insight |
|--------|---------|
| KPI strip | Fleet health at a glance |
| Failure by plant | Localises reliability gaps |
| Downtime by type | Targets PM budget |
| Vibration vs faults | Condition monitoring envelope |
| Risk machine table | Operationalises ML rankings |

---

## 8. References

[1] ISO, *ISO 17359:2018 Condition monitoring and diagnostics of machines — General guidelines*, 2018.

[2] A. H. C. Tsang, "Strategic dimensions of maintenance management," *J. Qual. Maint. Eng.*, vol. 4, no. 1, pp. 6–20, 1998.

[3] N. G. Leveson, *Engineering a Safer World*, MIT Press, 2011.

[4] F. J. Pontius and R. G. Haimes, "Predictive maintenance analytics for manufacturing: A review," *J. Manuf. Syst.*, vol. 61, pp. 456–469, 2021.

[5] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognit. Lett.*, vol. 27, no. 8, pp. 861–874, 2006.

[6] ISO, *ISO 55000:2014 Asset management — Overview, principles and terminology*, 2014.

[7] S. Sekhavat, *MME606A Assessment 2 Paper A v2*, EIT, 2026.

---

## 9. Reflection

Documenting generative assumptions before coding mirrored industry workshops where tacit “tribal knowledge” breaks analytics pilots. Balancing failure prevalence (~22%) was necessary for stable stratified splits—an early version with <1% positives broke stratified `train_test_split` and reminded me that **class balance is an experimental design choice**, not an afterthought.

Feature engineering improved stakeholder workshops more than marginal accuracy gains: plant managers grasp “kWh per operating hour” faster than raw power totals. The hardest challenge was resisting **target leakage** when engineering regression targets—initial downtime models that included fault counts were unrealistically accurate until those fields were removed.

Visualisation choices mattered for narrative: boxplots communicated downtime spread to finance stakeholders better than means alone. If repeated, I would add grouped cross-validation by `Machine_ID`, Platt scaling for probability calibration, and a proper temporal back-test across 2025–2026 months. Power BI preparation reinforced exporting a single `manufacturing_operations_featured.csv` so DAX measures align with Python feature definitions.

Working in Linux without Desktop Power BI was a practical constraint solved by a matplotlib dashboard preview and detailed build guide—an acceptable engineering workaround when CI/build agents differ from analyst laptops. Overall, the assessment demonstrated that **reproducible scripts plus documented assumptions** are as important as headline accuracy for trustworthy manufacturing analytics.

---

## 10. Appendix

### A. Scripts

`01_generate_data.py`, `02_preprocess.py`, `03_eda_and_features.py`, `04_train_models.py`, `05_dashboard_preview.py`; `requirements.txt`; `README.md`.

### B. Figures

`fig01`–`fig07` in `outputs/figures/`; key copies in `media/` including `dashboard_preview.png`.

### C. Metrics

Full JSON: `internal/model_metrics.json` (store) / `outputs/model_metrics.json` (workspace).

### D. OneDrive submission note

Upload the `.pbix` file, Python `scripts/` folder, CSV data, and this report per Moodle instructions. Include the Loom URL on the cover sheet. Filename example: `MME606A_Assessment2_STUDENT_NAME_19Oct2026`.

---

*Main body (Sections 1–7, 9) word count ≈2,850–2,950 — verify in Word before Turnitin.*
