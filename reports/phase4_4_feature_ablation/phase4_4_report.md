# Phase 4.4 Feature Ablation Study Report
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2)
- **Author:** AntiGravity (Implementation Agent)
- **Domain Lead / Owner:** Bright
- **Strategic Lead:** ChatGPT
- **Status:** Phase 4.4 Complete (Controlled Feature Ablation Study: Spindle Band Power vs. Locked Primary Set)

---

## Executive Summary

Phase 4.4 addresses the core research question established in Phase 3.2:

> **Research Question:** Does `smcAC_spindle_band_pwr` provide meaningful incremental predictive value beyond the locked 5-feature Primary Set?

To evaluate this hypothesis without introducing methodological bias:
- **Paired Grouped Validation:** Both feature sets were evaluated on the **exact same 16 Leave-One-Tool-Out (LOGO) folds** using identical model configurations and hyperparameter settings across four diverse model families (Ridge, SVR, Random Forest, Gradient Boosting).
- **Model A (Baseline):** Locked 5 Primary Features (`smcAC_rms`, `vib_spindle_kurtosis`, `vib_spindle_p2p`, `AE_table_rms`, `AE_spindle_p2p`).
- **Model B (Ablation Candidate Added):** 5 Primary Features + `smcAC_spindle_band_pwr`.

### Empirical Findings Summary:
1. **Negligible Overall Impact:** Across all 4 model families, the net change in out-of-fold MAE ranged between **$-0.0021\text{ mm}$ ($-1.47\%$)** and **$+0.0024\text{ mm}$ ($+1.59\%$)** — an absolute variation of less than **$2.5\text{ microns}$ ($0.0025\text{ mm}$)**. The observed MAE changes are small relative to the scale of VB measurement in this dataset.
2. **Inconsistent Across Model Families:**
   - **Ridge Regression:** Performance degraded slightly ($\text{MAE} = 0.1361 \rightarrow 0.1371\text{ mm}$, $+0.67\%$). 9 of 16 folds worsened.
   - **SVR (RBF):** Performance degraded ($\text{MAE} = 0.1481 \rightarrow 0.1505\text{ mm}$, $+1.59\%$; $R^2 = 0.3415 \rightarrow 0.3227$).
   - **Gradient Boosting:** Negligible change in MAE ($0.1381 \rightarrow 0.1389\text{ mm}$, $+0.59\%$), with a marginal improvement in RMSE ($0.1972 \rightarrow 0.1959\text{ mm}$) and $R^2$ ($0.4229 \rightarrow 0.4307$).
   - **Random Forest:** Marginal improvement ($\text{MAE} = 0.1412 \rightarrow 0.1392\text{ mm}$, $-1.47\%$), though its MAE remains inferior to 5-feature Ridge ($0.1361\text{ mm}$).
3. **Verdict:** This study falls squarely into **Case A (Little or inconsistent improvement)**:
   > `smcAC_spindle_band_pwr` is highly correlated with `smcAC_rms` but provides no compelling evidence of consistent incremental predictive value under the current LOGO evaluation.
4. **Strategic Decision:** **Retain the 5-feature Minimum Sufficient Set.** `smcAC_spindle_band_pwr` remains excluded from primary predictive models. High correlation with tool wear ($r_s = 0.756$) does not justify inclusion when a feature is $98.5\%$ collinear with an existing predictor (`smcAC_rms`).

---

## 1. Research Context & Purpose

In Phase 3.1, `smcAC_spindle_band_pwr` (spectral energy in the 11.0–16.5 Hz band capturing spindle rotational harmonics) exhibited a high rank correlation with measured flank wear ($r_s = 0.756$, within-tool median $r_s = 0.993$). However, in Phase 3.2, collinearity analysis revealed that `smcAC_spindle_band_pwr` shares a Pearson correlation of **$r = 0.9846$** with `smcAC_rms`.

In conventional machine learning pipelines, features with high individual target correlations are often indiscriminately combined. In accordance with this project's ethos:

> *"No favorite features. Keep only features that are necessary or meaningfully informative."*

Phase 4.4 was designed to test whether the physical interpretability of the $13.8\text{ Hz}$ kinematic harmonic translates into novel, non-redundant predictive signal for unseen tools, or whether it simply acts as a collinear surrogate for total motor current RMS.

---

## 2. Experimental Setup & Integrity Verification

The study was performed using the verified dataset:
📁 [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv)

### 2.1 Dataset Integrity Verification
- **Total Valid Observations:** Exactly **145 rows** (zero missing values in features or target).
- **Target:** **`VB_mm`** (continuous regression, range $0.00 - 1.53\text{ mm}$, mean $0.3394\text{ mm}$).
- **Validation Scheme:** **Leave-One-Group-Out (LOGO)** across all **16 independent tool cases (`case`)**.
- **Paired Validation Guarantee:** Folds, training subsets, and test subsets were identical between Model A and Model B. The comparison is strictly paired fold-by-fold.
- **Strict Leakage Barrier:** `case`, `run`, `cumulative_time_min`, and `condition_id` were excluded from model input matrices.

### 2.2 Feature Sets Under Test
```text
Model A (Locked Primary 5 Features):
├── smcAC_rms              (Current / Dynamic Load Proxy)
├── vib_spindle_kurtosis   (Vibration / Impact Peakedness)
├── vib_spindle_p2p        (Vibration / Dynamic Excursion Range)
├── AE_table_rms           (Acoustic-emission envelope associated with cutting/contact activity)
└── AE_spindle_p2p         (Acoustic Emission / Spindle Bursts)

Model B (Ablation Candidate Added - 6 Features):
└── Model A (5 Primary Features) + smcAC_spindle_band_pwr
```

### 2.3 Models & Hyperparameter Specifications
Identical hyperparameters were maintained from Phase 4.1–4.3 (zero tuning):
1. **Ridge Regression:** `alpha=1.0`, `StandardScaler` inside pipeline fitted strictly on training folds.
2. **Support Vector Regression (SVR):** `kernel='rbf'`, `C=1.0`, `epsilon=0.1`, `gamma='scale'`, `StandardScaler` inside pipeline.
3. **Random Forest Regressor:** `n_estimators=100`, `max_depth=5`, `min_samples_split=4`, `min_samples_leaf=2`, `random_state=42`.
4. **Gradient Boosting Regressor:** `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `random_state=42`.

---

## 3. Overall Out-of-Fold (OOF) Benchmark Comparison

The out-of-fold predictions across all 145 samples were aggregated to evaluate global regression performance:

| Model | Feature Set | MAE (mm) | RMSE (mm) | $R^2$ | $\Delta \text{MAE}$ (mm) | Rel. $\Delta \text{MAE}$ (%) | $\Delta \text{RMSE}$ (mm) | $\Delta R^2$ | Folds Improved | Strategic Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ridge** | Primary 5 (Model A) | **0.1361** | **0.1988** | **0.4139** | baseline | 0.0% | baseline | baseline | Baseline | Baseline Reference |
| **Ridge** | Primary 5 + Band Power (Model B) | 0.1371 | 0.1999 | 0.4072 | **+0.0009** | **+0.67%** | +0.0011 | -0.0067 | 5 / 16 (Worse: 9/16) | Negligible / Mild Degradation |
| **SVR** | Primary 5 (Model A) | **0.1481** | **0.2107** | **0.3415** | baseline | 0.0% | baseline | baseline | Baseline | Baseline Reference |
| **SVR** | Primary 5 + Band Power (Model B) | 0.1505 | 0.2137 | 0.3227 | **+0.0024** | **+1.59%** | +0.0030 | -0.0188 | 8 / 16 (Worse: 7/16) | Mild Degradation |
| **Random Forest** | Primary 5 (Model A) | 0.1412 | 0.2077 | 0.3604 | baseline | 0.0% | baseline | baseline | Baseline | Baseline Reference |
| **Random Forest** | Primary 5 + Band Power (Model B) | **0.1392** | **0.2056** | **0.3729** | **-0.0021** | **-1.47%** | -0.0021 | +0.0125 | 9 / 16 (Worse: 7/16) | Marginal Improvement |
| **Gradient Boosting**| Primary 5 (Model A) | **0.1381** | 0.1972 | 0.4229 | baseline | 0.0% | baseline | baseline | Baseline | Baseline Reference |
| **Gradient Boosting**| Primary 5 + Band Power (Model B) | 0.1389 | **0.1959** | **0.4307** | **+0.0008** | **+0.59%** | -0.0013 | +0.0078 | 9 / 16 (Worse: 6/16) | Negligible Change |

*Summary table serialized at:* [`reports/phase4_4_feature_ablation/feature_ablation_summary.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_4_feature_ablation/feature_ablation_summary.csv)  
*Paired predictions (145 rows $\times$ 15 cols):* [`reports/phase4_4_feature_ablation/oof_predictions_ablation.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_4_feature_ablation/oof_predictions_ablation.csv)

*Reference Visualization:*
- [Figure 2: Overall Out-of-Fold MAE Comparison: Model A vs. Model B](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_4_feature_ablation/figures/fig2_overall_mae_comparison.png)

---

## 4. Fold-Level Paired Analysis (Tool-by-Tool Stability)

To ensure conclusions were not skewed by global aggregation, we analyzed the paired fold-by-fold differences ($\Delta \text{MAE} = \text{MAE}_B - \text{MAE}_A$) across all 16 independent tool inserts:

### 4.1 Fold Stability Statistics

| Model | Mean $\Delta \text{MAE}$ (mm) | Median $\Delta \text{MAE}$ (mm) | Min $\Delta \text{MAE}$ (mm) | Max $\Delta \text{MAE}$ (mm) | Folds Improved ($\Delta < 0$) | Folds Worse ($\Delta > 0$) | Folds Tied ($\Delta = 0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | $+0.0007$ | $+0.0002$ | $-0.0005$ | $+0.0054$ | 5 / 16 (31.3%) | **9 / 16 (56.3%)** | 2 / 16 |
| **SVR** | $+0.0053$ | $+0.0008$ | $-0.0145$ | $+0.0537$ | 8 / 16 (50.0%) | 7 / 16 (43.8%) | 1 / 16 |
| **Random Forest** | $-0.0015$ | $-0.0009$ | $-0.0105$ | $+0.0053$ | 9 / 16 (56.3%) | 7 / 16 (43.8%) | 0 / 16 |
| **Gradient Boosting**| $-0.0020$ | $-0.0003$ | $-0.0164$ | $+0.0162$ | 9 / 16 (56.3%) | 6 / 16 (37.5%) | 1 / 16 |

*Fold-level dataset serialized at:* [`reports/phase4_4_feature_ablation/fold_level_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_4_feature_ablation/fold_level_comparison.csv)

*Reference Visualization:*
- [Figure 1: Paired Fold-Level $\Delta \text{MAE}$ Across 16 Folds](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_4_feature_ablation/figures/fig1_fold_level_delta_mae.png)

### 4.2 Key Fold-Level Observations:
1. **Ridge Performance:** In Ridge Regression, adding `smcAC_spindle_band_pwr` worsened prediction error in **9 out of 16 folds** ($\text{MAE} = 0.1361 \rightarrow 0.1371\text{ mm}$, $+0.67\%$). The degradation is consistent with the high collinearity between `smcAC_rms` and `smcAC_spindle_band_pwr` ($r = 0.985$), although the present experiment does not isolate the exact mechanism.
2. **SVR Sensitivity:** SVR degraded overall ($\text{MAE} = 0.1481 \rightarrow 0.1505\text{ mm}$, $+1.59\%$, worsening in 7 of 16 folds with a $+0.0537\text{ mm}$ error increase on Case 16). The observed degradation is consistent with the added redundant feature affecting the feature-space representation used by the RBF kernel, but the mechanism was not directly isolated in this experiment.
3. **Tree Models (Random Forest & Gradient Boosting):** For Gradient Boosting, overall MAE changed negligibly ($0.1381 \rightarrow 0.1389\text{ mm}$, $+0.59\%$, median fold $\Delta\text{MAE} = -0.0003\text{ mm}$). For Random Forest, MAE decreased slightly ($0.1412 \rightarrow 0.1392\text{ mm}$, $-1.47\%$). The near-zero overall change suggests that the tree-based models gained little additional predictive information from the added feature.

---

## 5. Methodological Transparency Statement

In accordance with rigorous data science governance:

> *"The five Primary Features were selected during Phase 3.1–3.2 using exploratory analysis that included the full valid dataset. Therefore, this feature selection was not completely blind to the target. Phase 4.4 does not use held-out test performance to reselect features; it evaluates the pre-locked candidate against a predefined ablation candidate under identical LOGO folds."*

No additional leakage was introduced during Phase 4.4. Both feature sets were pre-specified and evaluated using identical outer LOGO folds, with all model fitting performed strictly within each training fold.

---

## 6. Final Gate — Explicit Answers to Strategic Review Questions

### 1. Does `smcAC_spindle_band_pwr` improve predictive performance?
**No meaningful or consistent improvement is observed.**  
Across the four evaluated model families, the net change in MAE is between $-0.0021\text{ mm}$ ($-1.47\%$) and $+0.0024\text{ mm}$ ($+1.59\%$). For the linear benchmark leader (Ridge), performance degraded slightly ($+0.67\%$). For Gradient Boosting, MAE changed by less than $1\text{ micron}$ ($+0.0008\text{ mm}$).

### 2. Is the improvement consistent across LOGO folds?
**No.**  
In Ridge Regression, 9 out of 16 folds showed higher error with the feature added. In SVR, 7 folds worsened. In Gradient Boosting and Random Forest, improvements occurred in only 9 of 16 folds, while the remaining 7 folds were unchanged or degraded. The effect varies randomly around zero across tool inserts.

### 3. Is the improvement consistent across model families?
**No.**  
Ridge and SVR degraded in MAE ($+0.67\%$ and $+1.59\%$), Gradient Boosting exhibited virtually identical MAE ($+0.59\%$), and Random Forest showed a marginal decrease ($-1.47\%$). There is no consensus across model families that the feature adds value.

### 4. Should it remain outside the Primary Set or be considered for the final feature set?
**It must REMAIN OUTSIDE the Primary Set.**  
Under the strategic interpretation rules, this outcome represents **Case A (Little or inconsistent improvement)**. Adhering to the principle of a **Minimum Sufficient Feature Set**:
> `smcAC_spindle_band_pwr` is highly correlated with `smcAC_rms` but provides no compelling evidence of consistent incremental predictive value under the current LOGO evaluation. The five-feature Primary Set is therefore retained as the Minimum Sufficient Feature Set for subsequent evaluation.

### 5. What is the recommended next step?
The ablation study provides no compelling evidence that the additional feature adds consistent predictive value. The next priority is:
- **Phase 4.5 — Context Fusion Model & Cutting Condition Integration:** Test whether combining the 5 Primary Sensor Features with known machine operating parameters (`material_code`, `DOC_mm`, `feed_mm_rev`) resolves the performance discrepancy between Cast Iron ($\text{MAE} \approx 0.095\text{ mm}$) and Stainless Steel ($\text{MAE} \approx 0.22\text{ mm}$).
- Strictly avoid hyperparameter tuning until context integration is evaluated.

---

## Phase 4.4 Status: COMPLETE

*Report produced by AntiGravity (Implementation Agent) under the direction of Bright and Strategic Lead ChatGPT.*
