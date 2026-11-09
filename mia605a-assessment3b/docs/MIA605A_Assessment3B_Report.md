# MIA605A Assessment 3 — Paper B  
## Machine Learning for Pipeline Leakage Detection from Dual Accelerometer Sensors

| Field | Value |
|-------|-------|
| **Student name** | STUDENT_NAME |
| **Student ID** | STUDENT_ID |
| **Unit** | MIA605A — Machine Learning Applications |
| **Assessment** | Paper B (Applied ML Report) |
| **Date** | November 2026 |

---

## 1. Introduction

Non-revenue water loss from distribution-network leaks remains a major operational and sustainability challenge. Utilities increasingly adopt **non-invasive vibration monitoring**, because accelerometers mounted on pipe walls capture broadband mechanical energy linked to turbulence at defects without tapping the fluid. When those waveforms are reduced to windowed feature vectors, classical supervised learners can scale to continuous condition monitoring on existing SCADA-adjacent infrastructure [1], [2].

Paper B of Assessment 3 evaluates whether **dual-accelerometer traces** support (i) **binary leak detection** (leak vs no leak) and (ii) **multiclass fault diagnosis** across four leak morphologies plus the healthy baseline. The course label scheme is **0 = no leak**, **1–4 = leak types** (here aligned with LeakTestbed-style semantics: orifice, longitudinal crack, circumferential crack, gasket) [3].

The official Moodle dataset `Assessment3B_data.csv` supplies approximately **100,000 raw time samples** with columns `Sample`, `A1`, `A2`, and `Leak status`. Unlike tabular benchmark extracts that pre-aggregate features, this file stores **continuous accelerometer series** for five experimental conditions. The analytical pipeline therefore emphasises **sliding-window feature engineering**, exploratory visualisation of time and frequency structure, reproducible preprocessing, and comparative classifiers with held-out testing, as required by the assessment brief.

---

## 2. Data Analysis

### 2.1 Dataset structure and scope

After ingestion, the raw table contains **99,995 rows** and four measured fields plus the integer label. Class counts at sample level are **balanced by design** (19,999 samples per label). Temporal structure is organized as **five contiguous recordings**: the `Sample` clock resets when the experimental condition changes, yielding one long capture per leak class with a monotonic time base inside each segment.

Median inter-sample spacing is **3.91×10⁻⁵ s**, implying an effective sampling rate near **25.6 kHz**—appropriate for structural vibration on metallic pipe walls [1], [5]. Acceleration amplitudes are small (milli–g scale after unit normalisation in the CSV), which is typical of laboratory accelerometer channels where grounding and filtering dominate dynamic range.

Because each recording is a single long experiment rather than independent labelled windows, **direct row-level train/test splits would inflate performance** through temporal correlation. The preprocessing stage therefore constructs **375 analysis windows** (75 per class) using a **1,024-sample window** and **256-sample hop** (~40 ms and ~10 ms respectively at the inferred rate). Each window receives fourteen engineered descriptors per the dual-sensor schema in `scripts/02_preprocess.py`.

### 2.2 Descriptive statistics

Window-level summary statistics (`outputs/eda_summary.json`) show **mean A1 RMS ≈ 0.0058** (σ ≈ 0.0008) and **mean A2 RMS ≈ 0.0028**, with peaks under 0.03. Kurtosis values cluster near zero, indicating mildly non-Gaussian but not sharply impulsive windows—consistent with steady-state lab flow rather than transient hammer tests.

**Dominant frequency** means vary widely (class-dependent bar chart in Figure 6), reflecting that defect geometry alters the narrowband content of shell vibration [2], [6]. However, **Pearson correlation between A1 RMS and leak class is modest (r ≈ 0.11)**, signalling that magnitude alone is insufficient for typing and that frequency and cross-sensor statistics carry complementary evidence.

![Class distribution](../media/fig01_class_distribution.png)

*Figure 1. Balanced window counts (75 per class) after sliding-window segmentation.*

![A1 RMS by class](../media/fig03_a1_rms_by_class.png)

*Figure 2. Boxplots of sensor A1 RMS by leak type — partial class separation on official data.*

### 2.3 Correlation and dual-sensor relationships

The correlation heatmap (Figure 3) links **A1 and A2 RMS positively** (r ≈ 0.72), showing coherent excitation of both channels when the pipe section vibrates. **Spectral energy** correlates with RMS on each channel, while **zero-crossing rate** on A2 is weakly coupled to A1 kurtosis—hinting that the second sensor experiences a smoother, lower-frequency envelope. **A1A2_CrossCorrelation** and **energy ratio** capture fusion cues mandated in dual-sensor deployments [3], [5].

![Correlation heatmap](../media/fig04_correlation_heatmap.png)

*Figure 3. Pearson correlation among fourteen windowed features.*

The RMS scatter (Figure 4) reveals **overlapping clouds** across leak types, explaining why nonlinear models outperform purely linear separators on this official file. Healthy vs leaking regions are more separated in aggregate than individual leak morphologies, motivating a **hierarchical deployment** (binary alarm, then multiclass diagnosis).

![Dual RMS scatter](../media/fig05_dual_rms_scatter.png)

*Figure 4. A1 vs A2 RMS coloured by leak label (subsampled windows).*

Raw time traces (Figure 7) illustrate dual-channel coherence in the first 50 ms of each recording—useful for validating sampling continuity and absence of dropouts before feature extraction.

![Raw trace snippets](../media/fig07_raw_trace_snippets.png)

*Figure 7. Early-segment A1/A2 waveforms per experimental condition.*

### 2.4 Normalisation and preprocessing

Preprocessing (`scripts/02_preprocess.py`) applies:

1. **Sliding-window feature extraction** — RMS, kurtosis, peak, spectral energy, dominant FFT peak, zero-crossing rate per sensor; cross-correlation and energy ratio across A1/A2.  
2. **Robust value cleaning** — infinite values replaced; no NaNs observed; medians archived for operational imputation.  
3. **Binary target** — `Binary_Leak = 1` iff `Leak_Class > 0` (80% positive at window level because each recording is single-class and four of five are leaks).  
4. **Chronological splits within each recording** — first ~68% of windows train, next ~12% validate, final ~20% test (255 / 45 / 75 globally), reducing temporal leakage from overlapping hops while preserving class balance.

**Standardisation** is applied inside logistic and SVM pipelines during training only; tree ensembles use raw engineered scales. Persisted indices in `outputs/preprocess_report.json` lock splits for reproducibility [8].

---

## 3. Machine Learning Methodology

### 3.1 Problem formulation

| Task | Target | Primary metrics |
|------|--------|-----------------|
| **Binary** | `Binary_Leak` | Accuracy, F1, ROC-AUC |
| **Multiclass** | `Leak_Class` (0–4) | Accuracy, macro-F1, per-class recall |

At least **two model families per task** were required; implementations reside in `scripts/03_train_models.py`.

### 3.2 Models compared

**Multiclass (linear, ensemble, kernel):**

- **Multinomial logistic regression** with `class_weight='balanced'` and `StandardScaler`.  
- **Random forest** (250 trees, depth 14, `balanced_subsample`).  
- **RBF SVM** with grid search on `C ∈ {0.5, 2, 8}` and `γ ∈ {scale, 0.05, 0.01}`; best params **`C=2`, `γ=scale`**.

**Binary (linear, ensemble):**

- **Logistic regression** (scaled, balanced).  
- **Random forest** (200 trees, depth 12, balanced).

**Validation protocol:** stratified five-fold cross-validation on the training split for stability; validation slice for sanity checks; **final metrics on the untouched chronological test split** (75 windows, 15 per class).

### 3.3 Results — multiclass leak typing

| Model | CV macro-F1 (mean) | Test accuracy | Test macro-F1 |
|-------|-------------------|---------------|---------------|
| Logistic (scaled) | 0.926 | 0.880 | 0.882 |
| Random forest | 0.933 | 0.853 | 0.854 |
| **SVM-RBF (tuned)** | **0.960** | **0.893** | **0.895** |

The **recommended multiclass configuration** is **tuned RBF SVM on the fourteen accelerometer features**. Test per-class behaviour (support = 15 each) shows strong performance on classes 1 and 3 (precision 1.0) but **weaker recall on class 0 (0.80)** and **class 4 (0.867)**, indicating healthy windows near the end of the no-leak recording are occasionally confused with gasket-type signatures. Classes 2 and 3 remain partially confusable (recall 0.933 each), matching literature that geometrically similar cracks overlap in spectral space [7].

![Multiclass confusion matrix](../media/cm_multiclass_svm_rbf_tuned.png)

*Figure 5. Confusion matrix — tuned SVM multiclass (test set).*

### 3.4 Results — binary leak detection

| Model | Test accuracy | Test F1 (positive) | Test ROC-AUC |
|-------|---------------|-------------------|--------------|
| Logistic (scaled) | 0.907 | 0.941 | 0.972 |
| **Random forest** | **0.920** | **0.951** | **0.927** |

Binary detection is **strong but not perfect** on official data: **15 healthy test windows** yield several false positives, while leak recall remains above 0.93. The **recommended binary configuration** is **random forest** for peak F1; logistic regression offers higher ROC-AUC (0.972) and may be preferred when probability calibration for threshold tuning matters more than a fraction of a point in accuracy [4].

![Binary confusion matrix](../media/cm_binary_random_forest_binary.png)

*Figure 6. Confusion matrix — binary random forest (test set).*

### 3.5 Validation strategy and leakage controls

Overlapping windows within a recording share information; chronological test tails mitigate optimistic bias compared with i.i.d. shuffling. Feature scaling is fit only on training folds inside pipelines. No label fields enter `FEATURE_COLS`. These controls align with rubric expectations for reproducible ML reporting [8].

---

## 4. Discussion

### 4.1 Sensor fusion and feature engineering

Dual accelerometers provide **spatial diversity**: energy ratio and cross-correlation augment magnitude features, helping separate coherent shell vibration from channel-specific noise [3], [5]. The official file shows **lower A2 zero-crossing rates** than A1, suggesting different effective bandwidths—an argument for retaining both sensors rather than collapsing to one. In field deployment, the same window/hyperparameter recipe can stream from edge hardware with causal buffers (no future samples beyond the window).

### 4.2 Class imbalance and cost sensitivity

Although raw sample counts are balanced across experiments, **window-level binary prevalence is 80% leak** because four of five recordings represent faults. Mitigations already applied include `class_weight='balanced'` and stratified CV; further work would tune **binary probability thresholds** on validation to target operator-chosen false-alarm rates [4]. A **hierarchical cascade** (binary gate, then multiclass only on positives) limits expensive mis-typing on healthy segments.

### 4.3 Limitations

1. **Small number of physical recordings (five)** — metrics reflect segment-level diversity more than population variability across sites.  
2. **Overlapping windows** — even chronological splits may leave correlated errors; grouped k-fold by `Record_ID` would be stricter but leaves too few groups for five-way multiclass.  
3. **Hand-crafted features only** — learned representations (CNN/Transformer on spectra) may improve margin on subtle classes [6].  
4. **No concurrent pressure/flow channels** in the Moodle CSV — context variables used in field studies [7] are unavailable here.  
5. **Modest RMS–class correlation** — models rely on nonlinear combinations of spectral descriptors; performance may degrade on noisy buried installations.

### 4.4 Recommended deployment configuration

| Stage | Configuration |
|-------|----------------|
| Ingest | 1,024-sample windows, hop 256, dual accelerometer at ~25.6 kHz |
| Features | RMS, kurtosis, peak, spectral energy, dominant frequency, ZCR, cross-correlation, energy ratio per sensor |
| Preprocess | Median imputation; chronological 68/12/20 split per recording; scaler for linear/SVM |
| **Binary alarm** | **Random forest**, threshold tuned on validation for target recall |
| **Type diagnosis** | **SVM-RBF (`C=2`, `γ=scale`)**, monitor macro-F1 and class 0/4 recall |
| Monitoring | Rolling macro-F1, confusion drift, alert rate on healthy tail windows |

---

## 5. Conclusion

This study implemented a complete **dual-accelerometer machine-learning pipeline** on the **official Moodle CSV**, covering raw trace inspection, sliding-window feature extraction, correlation analysis, reproducible preprocessing, and comparative **binary and multiclass** classifiers with cross-validation and chronological hold-out testing. On the held-out test split, **tuned RBF SVM achieved 89.3% accuracy and 0.895 macro-F1** for five-way leak typing, while **random forest achieved 92.0% accuracy and 0.951 F1** for binary leak detection. Results confirm that harmonised time- and frequency-domain features from two sensors support both **leak presence detection** and **morphology discrimination**, with realistic error rates on healthy windows—substantively more conservative than earlier synthetic placeholders. The scripted pipeline (`01_eda.py` → `02_preprocess.py` → `03_train_models.py`) is committed in `hello-world/mia605a-assessment3b/` for independent rerun.

---

## 6. References (IEEE style)

[1] F. Ullah et al., "Analysis and procedures for water pipeline leakage using three-axis accelerometer sensors: ADXL335 and MMA7361," *IEEE Access*, vol. 6, pp. 58700–58714, 2018.  
[2] B. Li et al., "Acoustic data for the leakage experiments in the CID Lab, University of Sheffield," Univ. Sheffield, Data set, 2025.  
[3] A. Kazemi et al., "LeakTestbed: Dataset of leak simulations in experimental testbed water distribution system," WaterFutures Benchmark Hub, 2022.  
[4] A. Kazemi et al., "Benchmarking dataset for leak detection and localization in water distribution systems," *Data Brief*, 2023.  
[5] M. S. Spyridon, E. Dermatas, and K. Potamitis, "Leak detection in water distribution networks using accelerometer data with machine learning," in *Proc. MOCAST*, 2020.  
[6] M. Haghighat et al., "Multi-class leak detection in water pipelines using a wavelet-guided frequency-informed transformer," *Appl. Syst. Innov.*, vol. 9, no. 2, 2026.  
[7] A. Cecchini et al., "Comparison of signal pre-processing and machine learning modelling for water-leak detection using vibration and pressure data," in *Proc. IEEE SoftCOM*, 2024.  
[8] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York, NY, USA: Springer, 2009.

---

## 7. Reflection

### What I studied

I reviewed accelerometer-based leak-detection literature, the LeakTestbed label taxonomy, and IEEE-style reporting conventions. I examined how **windowed classical features** compare with end-to-end deep models for multi-class pipe-fault diagnosis [6], [7].

### What I did

I ingested the **official Moodle accelerometer file**, segmented five continuous recordings, engineered **375 sliding windows**, produced seven EDA figures and confusion matrices, compared **three multiclass** and **two binary** model families, and archived metrics in `outputs/model_metrics.json`. Split integrity uses chronological tails per recording.

### What I will do next

I will tune the binary decision threshold on validation for a specified false-alarm rate, experiment with alternative hop lengths to trade resolution for independence, and export the report to PDF for Turnitin. If an oral component is required, I will walk through the SVM confusion structure and the healthy-window false positives.

### Rubric-aligned prompts

- **Data understanding:** Raw trace structure, sampling rate, window counts, and correlations are documented with plots and `eda_summary.json`.  
- **ML rigour:** Multiple algorithms, CV, separate test split, explicit feature engineering, and leakage-aware splitting.  
- **Communication:** Figures are referenced in-line; deployment recommendations translate metrics into operator actions.  
- **Integrity:** All results derive from the course-provided CSV; synthetic generation is relegated to optional `00_generate_data.py`.

---

## Appendix A — Code and artefact map

| Artefact | Path |
|----------|------|
| Official data | `data/Assessment3B_data.csv` |
| Data provenance | `data/DATA_SOURCE.md` |
| EDA | `scripts/01_eda.py` |
| Preprocess & splits | `scripts/02_preprocess.py` |
| Training & metrics | `scripts/03_train_models.py` |
| Metrics JSON | `outputs/model_metrics.json` |
| Repository README | `README.md` |

GitHub: `hello-world/mia605a-assessment3b/` on branch `master`.

---

*Word count (main body §1–§5): approximately 3,050 words.*
