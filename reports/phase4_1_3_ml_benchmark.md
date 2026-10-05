# Phase 4.1–4.3 Initial Machine Learning Benchmark Report
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2)
- **Author:** AntiGravity (Implementation Agent)
- **Domain Lead / Owner:** Bright
- **Strategic Lead:** ChatGPT
- **Status:** Phase 4.1–4.3 Complete (Data Preparation, Grouped Validation Protocol, Initial 5-Model Benchmark)

---

## Executive Summary

Phase 4.1–4.3 establishes the **initial Machine Learning benchmark** for tool flank wear ($V_B$) prediction using the **locked 5-feature primary candidate set** from Phase 3.2. 

In strict adherence to the project's manufacturing-first principles:
- **No additional leakage was introduced during Phase 4 model evaluation:** The model was evaluated exclusively via **Grouped Leave-One-Tool-Out (LOGO)** cross-validation across all **16 independent tool inserts (`case`)**. A tool insert never appeared in both training and test sets within any fold. The candidate feature set was locked prior to Phase 4 (though, as transparently documented, initial candidate screening in Phase 3.1–3.2 utilized dataset-level wear correlations).
- **No synthetic feature zoo or extra models:** Only the 5 locked primary sensor features (`smcAC_rms`, `vib_spindle_kurtosis`, `vib_spindle_p2p`, `AE_table_rms`, `AE_spindle_p2p`) were used.
- **Fair comparison across diverse model families:** A Mean Baseline, a regularized linear model (Ridge), a kernel method (SVR), and two tree ensemble algorithms (Random Forest, Gradient Boosting) were benchmarked on identical outer folds without aggressive hyperparameter tuning.
- **Key Finding:** All four ML models outperformed the mean baseline in the LOGO evaluation, reducing out-of-fold MAE from **$0.1998\text{ mm}$ (baseline)** down to **$0.136 - 0.148\text{ mm}$** (a $26 - 32\%$ error reduction). **Ridge Regression ($\text{MAE} = 0.1361\text{ mm}$)** and **Gradient Boosting Regressor ($\text{MAE} = 0.1381\text{ mm}$, $R^2 = 0.4229$, lowest fold std $\pm 0.0772\text{ mm}$)** emerged as the two strongest initial benchmark candidates, exhibiting complementary strengths across the evaluation metrics.

---

## 1. Input Dataset & Primary Features

The benchmark was executed directly on the finalized Phase 3.2 feature dataset:
📁 [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv)

### 1.1 Five Locked Primary Features
The model input matrix $X$ was strictly restricted to the 5 primary candidate features:
1. **`smcAC_rms`** (Current / Dynamic Load): Spindle motor current RMS / dynamic load proxy ($0.86 - 3.47\text{ V}$).
2. **`vib_spindle_kurtosis`** (Vibration / Impact Peakedness): Captures changes in the peakedness of spindle-side vibration envelope associated with increasing wear ($-1.17 - 3093.8$).
3. **`vib_spindle_p2p`** (Vibration / Dynamic Excursion): Spindle vibration peak envelope excursion range ($0.10 - 2.25\text{ V}$).
4. **`AE_table_rms`** (Acoustic Emission / Friction): Acoustic-emission envelope associated with cutting/contact activity and showing strong within-tool association with measured $V_B$ ($0.02 - 0.35\text{ V}$).
5. **`AE_spindle_p2p`** (Acoustic Emission / Spindle Bursts): Range of the processed acoustic-emission envelope measured at the spindle-side sensor ($0.09 - 1.18\text{ V}$).

### 1.2 Target Variable
- **`VB_mm`** (Continuous Flank Wear in mm): Ranging from $0.00\text{ mm}$ (fresh insert) to $1.53\text{ mm}$ (severely degraded insert), with a sample mean of $0.3394\text{ mm}$ and standard deviation of $0.2605\text{ mm}$.

### 1.3 Forbidden Model Inputs (Strictly Partitioned into Metadata)
- **`case`:** Categorical tool ID (used exclusively as the grouping key for LOGO cross-validation).
- **`run`:** Sequence pass counter (prohibited; represents pass ordering).
- **`cumulative_time_min`:** Prohibited; acts as a **target proxy / tool-age leakage**.
- **`condition_id`:** Metadata grouping key for future unseen-condition validation.

---

## 2. Phase 4.1: ML Dataset Preparation & Integrity Audit

Before running cross-validation, a rigorous data-quality audit was executed on the input matrix:
- **Usable Rows:** Exactly **145 rows** (100% of the verified valid cutting runs).
- **Missing Values:** Exactly **0 missing values (0.0%)** across all 5 primary features and the target.
- **Duplicate Rows:** **0 duplicate rows** on `(case, run)`.
- **Data Types:** All 5 predictive features and the target are verified 64-bit floating-point numeric arrays (`float64`).
- **Variance Check:** All 5 features exhibit healthy non-zero variance (minimum variance $= 0.00337$ in `AE_table_rms`).
- **Grouping Integrity:** Exactly **16 distinct tool cases**, with run counts ranging from 1 run (Case 6, aborted) to 20 runs (Case 11).

*Audit Result:* Dataset preparation passed 100% with zero data modifications required.

---

## 3. Phase 4.2: Validation Protocol — Grouped Leave-One-Tool-Out (LOGO)

### 3.1 Why Random Splitting Was Avoided
In machining analytics, multiple cutting passes are performed sequentially on the same physical tool insert until tool failure. If standard random train/test splits or ordinary K-Fold cross-validation were used:
- Passes from the **same tool insert** would appear in both training and test folds.
- The model would learn the idiosyncratic baseline voltage offsets or mounting runout of specific tool inserts rather than the physical degradation physics.
- This creates **optimistic data leakage**, producing falsely low test errors that collapse when deployed on a new machine setup.

### 3.2 The Grouped Leave-One-Tool-Out Architecture
To simulate genuine industrial tool replacement:
- We implemented **`LeaveOneGroupOut` (16 Folds)** using `case` (tool ID) as the grouping factor.
- In each fold, the model was trained on 15 independent tools ($125 - 144$ runs) and tested on the 1 held-out tool ($1 - 20$ runs).
- **Core Evaluation Question:** *"Can the sensor-driven model predict flank wear for a brand new tool insert that it has never encountered during training?"*

### 3.3 Preprocessing Leakage Prevention
For models requiring feature standardization (Ridge Regression and SVR):
- The `StandardScaler` was placed inside a scikit-learn `Pipeline` and **fitted strictly on the training partition of each fold**.
- Test fold features were transformed using the training fold's mean and standard deviation. No global dataset statistics leaked into the test partitions.
- Tree-based models (Random Forest, Gradient Boosting) were trained on raw unscaled features.

---

## 4. Phase 4.3: Model Benchmark Results

Five models representing distinct learning paradigms were benchmarked on identical outer folds:
1. **Model 0 — Mean Baseline:** Predicts the training set mean $\overline{y}_{\text{train}}$ for all test observations (the minimum reference threshold).
2. **Model 1 — Ridge Regression:** L2-regularized linear model (`alpha=1.0`), scaled inputs.
3. **Model 2 — Support Vector Regression (SVR):** Non-linear kernel regression (`kernel='rbf'`, `C=1.0`, `epsilon=0.1`), scaled inputs.
4. **Model 3 — Random Forest Regressor:** Bagged ensemble (`n_estimators=100`, `max_depth=5`, `min_samples_split=4`, `min_samples_leaf=2`, `random_state=42`).
5. **Model 4 — Gradient Boosting Regressor:** Boosted ensemble (`n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `random_state=42`).

### 4.1 Overall Out-of-Fold (OOF) Performance Summary

| Model | MAE (mm) | RMSE (mm) | $R^2$ | Fold MAE Mean (mm) | Fold MAE Std (mm) | Fold MAE Min (mm) | Fold MAE Max (mm) | Status / Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mean Baseline** | 0.1998 | 0.2644 | -0.0367 | 0.2035 | $\pm 0.0838$ | 0.1179 | 0.4246 | Reference benchmark (predicts fold mean) |
| **Ridge Regression** | **0.1361** | 0.1988 | 0.4139 | 0.1546 | $\pm 0.1005$ | **0.0447** | 0.3622 | **Lowest overall MAE** ($31.9\%$ error reduction vs baseline) |
| **Gradient Boosting**| 0.1381 | **0.1972** | **0.4229** | 0.1452 | **$\pm 0.0772$** | 0.0590 | **0.3600** | **Highest $R^2$**, lowest RMSE, **most stable across folds** |
| **Random Forest** | 0.1412 | 0.2077 | 0.3604 | **0.1407** | $\pm 0.0836$ | 0.0605 | 0.3875 | Lowest unweighted fold average MAE |
| **SVR (RBF)** | 0.1481 | 0.2107 | 0.3415 | 0.1581 | $\pm 0.0843$ | 0.0638 | 0.3714 | Solid improvement; slightly conservative predictions |

*Data serialized at:* [`reports/phase4_model_summary.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_model_summary.csv)  
*Full predictions (725 rows):* [`data/phase4_oof_predictions.csv`](file:///D:/Project/Manufacturing%20Analytics/data/phase4_oof_predictions.csv)

*Reference Visualizations:*
- [Figure 1: Actual vs. Predicted VB Across Initial Benchmark Models](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase4_benchmark/fig1_actual_vs_predicted_by_model.png)
- [Figure 2: Residual Analysis Across Flank Wear Degradation Range](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase4_benchmark/fig2_residuals_vs_actual_vb.png)
- [Figure 3: Model MAE & RMSE Comparison Bar Chart](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase4_benchmark/fig3_model_mae_comparison.png)
- [Figure 4: Tool-to-Tool Fold MAE Stability Across 16 Independent Tool Folds](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase4_benchmark/fig4_fold_level_mae_variation.png)

---

## 5. Detailed Model Performance Analysis

### 5.1 Ridge Regression & Gradient Boosting: Two Strongest Benchmark Candidates
Instead of declaring a single winner, the benchmark reveals that **Ridge Regression and Gradient Boosting are the two strongest initial benchmark candidates, exhibiting complementary strengths across the evaluation metrics**:
- **Ridge Regression** achieves the **lowest overall out-of-fold MAE ($0.1361\text{ mm}$)**. Because the 5 primary features were pre-screened to ensure monotonic alignment with wear, a regularized linear hyperplane fits the global trend with high parameter efficiency. It avoids overfitting the small sample size ($N=145$).
- **Gradient Boosting Regressor** achieves the **lowest RMSE ($0.1972\text{ mm}$)**, the **highest $R^2$ ($0.4229$)**, and the **lowest fold-to-fold standard deviation ($\pm 0.0772\text{ mm}$)**. It provides slightly more stable predictions across diverse tool folds.
- **Fitted Ridge Coefficients (Standardized):**
  - `smcAC_rms`: $+0.1814$ (strongest positive driver)
  - `AE_table_rms`: $-0.0806$
  - `vib_spindle_kurtosis`: $+0.0746$
  - `AE_spindle_p2p`: $+0.0701$
  - `vib_spindle_p2p`: $+0.0065$
- **Gradient Boosting Feature Importances:**
  - `smcAC_rms`: $40.1\%$
  - `vib_spindle_kurtosis`: $19.9\%$
  - `AE_spindle_p2p`: $18.2\%$
  - `AE_table_rms`: $15.6\%$
  - `vib_spindle_p2p`: $6.2\%$
- **Observation:** Both models distribute predictive weight across all physical domains: spindle current load, acoustic emission contact dynamics, and vibration peakedness all contribute meaningfully.

### 5.2 Random Forest Regressor
- **Performance:** Random Forest achieved an overall MAE of **$0.1412\text{ mm}$** ($R^2 = 0.3604$) and the lowest unweighted mean fold MAE ($0.1407\text{ mm}$).
- **Feature Importances:** Highly consistent with Gradient Boosting (`smcAC_rms` $48.5\%$, `vib_spindle_kurtosis` $18.5\%$, `AE_table_rms` $16.3\%$, `AE_spindle_p2p` $13.0\%$).
- **Limitation:** Like all decision tree ensembles, Random Forest cannot extrapolate beyond the maximum target value seen in the training folds, leading to mild underprediction on extreme wear runs ($V_B > 0.60\text{ mm}$).

### 5.3 Support Vector Regression (SVR)
- **Performance:** SVR with an RBF kernel achieved **MAE $= 0.1481\text{ mm}$** and $R^2 = 0.3415$.
- **Behavior:** The default RBF kernel parameterization ($\gamma = \text{scale}$, $C=1.0$, $\epsilon=0.1$) acts conservatively, compressing predictions toward the median wear value. While it avoids large catastrophic errors, it yields higher residuals on fresh inserts ($V_B < 0.10\text{ mm}$) and severely worn inserts.

---

## 6. Tool-to-Tool Fold Stability & Material Stratification

A critical strength of grouped cross-validation is exposing how models behave across different workpiece materials and cutting conditions. Full fold-level metrics are recorded in [`reports/phase4_fold_metrics.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_fold_metrics.csv):

### 6.1 Performance by Workpiece Material

| Material | Number of Runs | Baseline MAE (mm) | Ridge MAE (mm) | Gradient Boosting MAE (mm) | Random Forest MAE (mm) | SVR MAE (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cast Iron** | 97 | 0.1674 | **0.0975** | **0.0945** | 0.0992 | 0.1094 |
| **Stainless Steel J45**| 48 | 0.2652 | **0.2142** | 0.2263 | 0.2262 | 0.2263 |

### Key Fold Insights:
1. **High Precision on Cast Iron:** On Cast Iron tools (e.g., Case 2, 3, 4, 10, 11, 12), all models achieve sub-$0.10\text{ mm}$ error. For Case 2, Ridge achieves an extraordinary **$\text{MAE} = 0.0447\text{ mm}$**; for Case 10, Gradient Boosting achieves **$\text{MAE} = 0.0590\text{ mm}$**. The progressive wear curves on Cast Iron are smooth and predictable.
2. **Higher Volatility on Stainless Steel:** Errors on Stainless Steel are roughly double those on Cast Iron ($\text{MAE} \approx 0.21 - 0.23\text{ mm}$), associated with higher cutting signal volatility and greater wear-rate variability observed on Stainless Steel J45 in this benchmark:
   - Case 13 experienced severe insert degradation, reaching $V_B = 1.53\text{ mm}$ (substantially higher than any other tool in the benchmark). Consequently, all models underpredicted Case 13 (Fold MAE $= 0.36 - 0.38\text{ mm}$).
3. **Single-Run Aborted Tool Case (Case 6):** Case 6 contains only 1 valid cut ($N=1$, Run 1, $V_B = 0.00\text{ mm}$, Stainless Steel J45, DOC 1.50 mm, Feed 0.25 mm/rev). Because the experiment was aborted after this initial pass, it represents a single fresh-tool observation rather than a multi-pass wear trajectory. Tree models predicted closer to the fresh baseline state (Random Forest: $0.119\text{ mm}$, Gradient Boosting: $0.166\text{ mm}$), while linear models (Ridge: $0.353\text{ mm}$) predicted higher due to the elevated baseline current under DOC 1.50 mm.

---

## 7. Residual & Bias Diagnostics

Analyzing the out-of-fold residuals ($e_i = y_i - \hat{y}_i$) across wear degradation stages:

```
                            Residual Bias by Wear Stage
Wear Degradation Stage       | Sample Count | Ridge Mean Error (mm) | GBDT Mean Error (mm) | Observational Notes
-----------------------------|:------------:|:---------------------:|:--------------------:|-----------------------------------
Fresh Insert (VB < 0.20 mm)  |   58 runs    |       -0.0381         |       -0.0412        | Slight overprediction of brand new tools
Moderate Wear (0.20-0.40 mm) |   43 runs    |       -0.0163         |       -0.0245        | Well-centered predictions
Severe Wear (VB >= 0.40 mm)  |   44 runs    |       +0.0940         |       +0.1243        | Moderate underprediction of severe wear
```

- **Underprediction at High Wear:** As observed across all models, when flank wear exceeds $0.40\text{ mm}$, predictions exhibit a positive residual bias ($+0.09$ to $+0.12\text{ mm}$). This occurs because runs with extreme wear ($V_B > 0.50\text{ mm}$) represent a minority of the dataset (only 18 runs), pulling regression models toward the global median.
- Ridge showed lower high-wear bias than Gradient Boosting in this evaluation ($+0.094\text{ mm}$ vs $+0.124\text{ mm}$ for GBDT).

---

## 8. Methodological Constraints & Transparency

In alignment with Phase 3.2 guidelines:
1. **Locked Exploratory Candidate Set:** We reiterate that the 5 primary features were selected during Phase 3.1–3.2 using dataset-level correlation checks and within-tool consistency audits. Therefore, we do not claim that the overall feature engineering was a completely blind out-of-fold feature selection.
2. **No Additional Leakage Introduced:** **No additional leakage was introduced during Phase 4 model evaluation.** All scaling and training steps occurred strictly within outer training folds.
3. **Zero Post-Hoc Tuning:** The 5 primary features were strictly locked prior to Phase 4.1. No features were adjusted, added, or removed after inspecting the cross-validation test scores.
4. **No Model Favoritism:** No hyperparameter tuning was conducted to favor any algorithm. Default/conservative configurations were evaluated fairly across all models.

---

## 9. Final Gate — Explicit Deliverable Answers

### A. Data Summary
- **Total Rows Used:** Exactly **145 valid cutting runs** (100% complete, zero missing values).
- **Features Used:** Exactly **5 Primary Sensor Features** (`smcAC_rms`, `vib_spindle_kurtosis`, `vib_spindle_p2p`, `AE_table_rms`, `AE_spindle_p2p`).
- **Target:** **`VB_mm`** (continuous regression).
- **Validation Groups:** **16 independent tool cases** evaluated via Grouped Leave-One-Tool-Out cross-validation.

### B. Model Results Table

| Model | MAE (mm) | RMSE (mm) | $R^2$ |
| :--- | :---: | :---: | :---: |
| **Mean Baseline** | 0.1998 | 0.2644 | -0.0367 |
| **Ridge Regression** | **0.1361** | 0.1988 | 0.4139 |
| **Gradient Boosting** | 0.1381 | **0.1972** | **0.4229** |
| **Random Forest** | 0.1412 | 0.2077 | 0.3604 |
| **Support Vector Regression (SVR)** | 0.1481 | 0.2107 | 0.3415 |

### C. Fold Stability Analysis
- **Fold MAE Mean & Range:**
  - Ridge: Mean $= 0.1546\text{ mm}$, Std $= \pm 0.1005\text{ mm}$, Range: $[0.0447, 0.3622]\text{ mm}$
  - Gradient Boosting: Mean $= 0.1452\text{ mm}$, **Std $= \pm 0.0772\text{ mm}$ (Lowest Variability)**, Range: $[0.0590, 0.3600]\text{ mm}$
  - Random Forest: Mean $= 0.1407\text{ mm}$, Std $= \pm 0.0836\text{ mm}$, Range: $[0.0605, 0.3875]\text{ mm}$
  - SVR: Mean $= 0.1581\text{ mm}$, Std $= \pm 0.0843\text{ mm}$, Range: $[0.0638, 0.3714]\text{ mm}$
  - Baseline: Mean $= 0.2035\text{ mm}$, Std $= \pm 0.0838\text{ mm}$, Range: $[0.1179, 0.4246]\text{ mm}$

### D. Problems & Unexpected Issues
- **None.** All 145 rows executed cleanly without pipeline errors or missing predictions. 
- Case 6 ($N=1$, Run 1, $V_B = 0.00\text{ mm}$) was verified directly from the dataset as a single-run aborted tool case and evaluated without fold degeneration.

### E. Proposed Next Step: Phase 4.4 — Feature Ablation Study
Rather than rushing into model hyperparameter tuning, the immediate next question is to establish feature necessity:
- **Ablation Experiment:** Test `smcAC_spindle_band_pwr` (which correlates at $r = 0.985$ with `smcAC_rms`) on the **exact same 16 LOGO folds**:
  $$\text{Model A (5 Primary Features)} \quad \text{vs.} \quad \text{Model B (5 Primary + } \text{smcAC\_spindle\_band\_pwr)}$$
- **Decision Criterion:**
  - If MAE / RMSE / $R^2$ do not improve meaningfully and consistently across models and folds $\rightarrow$ prune `smcAC_spindle_band_pwr` as redundant.
  - If performance improves consistently $\rightarrow$ retain it.
- This directly answers: *"Is this feature genuinely necessary, or merely highly correlated with a feature we already have?"*

---
*Report produced by AntiGravity (Implementation Agent) under the direction of Bright and Strategic Lead ChatGPT.*
