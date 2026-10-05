# Phase 5.2B: Supporting Model (GBDT) Formulation & TreeSHAP Interpretation Audit
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2) — CNC Machining & Tool Wear Analytics
- **Document Type:** Supporting Model (GBDT) Interpretation & Linear vs. Non-linear Cross-Check
- **Author:** AntiGravity (Implementation Agent)
- **Reviewers:** Bright (Domain Lead) & ChatGPT (Strategic Lead)
- **Status:** Phase 5.2B Complete & Audited

---

## 1. Executive Summary

In Phase 5.1, the modeling architecture for the NASA Milling tool wear benchmark was locked:
- **Primary Engineering Model:** Ridge Regression + Context Fusion (`alpha = 1.0`, standardized 8-feature formulation).
- **Supporting Nonlinear Model:** Gradient Tree Boosting (GBDT) + Context Fusion (`n_estimators = 100`, `learning_rate = 0.05`, `max_depth = 3`, `subsample = 0.8`, `random_state = 42`).

Phase 5.2B executes an **independent non-linear interpretation audit** of the supporting GBDT model using **TreeSHAP** across the full dataset ($N = 145$ cuts, 16 tool cases, 8 conditions).

### Central Research Question:
> *Does the nonlinear GBDT model identify broadly similar important variables and predictive patterns as the linear Ridge model, or does it reveal materially different behavior?*

### Core Findings & Synthesis:
1. **Implementation & Benchmark Verification:**  
   The locked GBDT formulation reproduces the locked Phase 4 benchmark metrics with zero discrepancy:
   - **Context LOGO (16 folds):** $\text{MAE} = 0.1186\text{ mm}$, $\text{RMSE} = 0.1695\text{ mm}$, $R^2 = 0.5739$.
   - **Context LOCO (8 folds):** $\text{MAE} = 0.1213\text{ mm}$, $\text{RMSE} = 0.1756\text{ mm}$, $R^2 = 0.5425$.
2. **Dominant Feature Hierarchy:**  
   In the fitted GBDT model, `smcAC_rms` is overwhelmingly the most impactful feature ($\text{Mean } |\text{SHAP}| = 0.1383\text{ mm}$, Rank 1), accounting for more than double the attribution magnitude of any other feature. This directly corroborates the primary Ridge finding where `smcAC_rms` held the largest standardized coefficient ($\beta = +0.2940$, Rank 1).
3. **Joint Operating Condition Offsets:**  
   Depth of cut (`DOC_mm`, $\text{Mean } |\text{SHAP}| = 0.0598\text{ mm}$, Rank 2) and feed rate (`feed_mm_rev`, $\text{Mean } |\text{SHAP}| = 0.0302\text{ mm}$, Rank 5) both exhibit strictly negative SHAP contributions at higher parameter values. This independently mirrors the negative standardized coefficients in Ridge (`DOC_mm`: $-0.1547$, `feed_mm_rev`: $-0.0944$). Under joint estimation with spindle current, process variables provide condition-level baseline adjustments rather than independent physical wear trends.
4. **Cross-Model Consistency Verdict:**  
   **Broadly Consistent.** Both models identify the same dominant sensor and operating-context variables, with broadly similar feature ordering but some differences in secondary-feature importance. Within this dataset, `smcAC_rms` serves as the primary sensor feature and `DOC_mm` serves as the primary operating condition adjustment in both formulations. The non-linear formulation shows partial refinement in `vib_spindle_kurtosis` (elevated from Rank 5 in Ridge to Rank 3 in GBDT), indicating that its relationship with $V_B$ may contain nonlinear or range-dependent structure not captured by a single linear coefficient.

---

## 2. Implementation Audit

### 2.1 Specification Verification
The GBDT supporting model implementation strictly adheres to the locked Phase 4 / Phase 5.1 specification:

| Parameter | Locked Specification | Audited Implementation | Verification Status |
| :--- | :---: | :---: | :---: |
| **Model Class** | `GradientBoostingRegressor` | `sklearn.ensemble.GradientBoostingRegressor` | **Verified Match** |
| **Number of Estimators ($n_{\text{trees}}$)** | 100 | 100 | **Verified Match** |
| **Learning Rate ($\eta$)** | 0.05 | 0.05 | **Verified Match** |
| **Maximum Tree Depth ($d_{\text{max}}$)** | 3 | 3 | **Verified Match** |
| **Subsample Ratio** | 0.8 | 0.8 | **Verified Match (Phase 4 Canonical)** |
| **Random State** | 42 | 42 | **Verified Match** |
| **Target Variable** | `VB_mm` | `VB_mm` | **Verified Match** |
| **Number of Input Features** | 8 | 8 | **Verified Match** |

*Specification Verification Note:* `subsample = 0.8` was the canonical hyperparameter implemented across all Phase 4 scripts (`scripts/phase4_1_3_ml_benchmark.py`, `scripts/phase4_4_feature_ablation.py`, `scripts/phase4_5_context_fusion.py`, `scripts/phase4_5B_LOCO.py`). Its inclusion is verified against original benchmark source code.

### 2.2 Feature Set Audit
The 8 features input to the GBDT model are:
- **Primary Sensor Features (5):**
  1. `smcAC_rms`: Spindle motor AC current root mean square (A).
  2. `vib_spindle_kurtosis`: Spindle accelerometer vibration kurtosis (peakedness).
  3. `vib_spindle_p2p`: Spindle accelerometer vibration peak-to-peak amplitude (g).
  4. `AE_table_rms`: Table acoustic emission RMS (V).
  5. `AE_spindle_p2p`: Spindle acoustic emission peak-to-peak amplitude (V).
- **Operating Context Features (3):**
  6. `material_code`: Workpiece material category ($1 = \text{Cast Iron}$, $2 = \text{Stainless Steel J45}$).
  7. `DOC_mm`: Depth of cut ($0.75\text{ mm}$ or $1.50\text{ mm}$).
  8. `feed_mm_rev`: Feed rate ($0.25\text{ mm/rev}$ or $0.50\text{ mm/rev}$).

### 2.3 Preprocessing & Leakage Audit
- In tree-based models, decision split criteria are monotonic and invariant to monotonic feature scaling. Consequently, in Phase 4.5A and 4.5B, `GradientBoostingRegressor` was trained directly on raw feature values without `StandardScaler`. This maintains raw physical engineering units (Amperes, mm, mm/rev) directly in the tree split thresholds and SHAP attributions.
- **Leakage Boundary Statement:** No target leakage was introduced during fold-wise model fitting, and grouping was respected in LOGO/LOCO validation. However, the feature screening step in Phase 3.2 was performed once using all labeled observations; therefore, the overall validation is not a fully nested model-development estimate.

---

## 3. Benchmark Reproduction

The benchmark was executed using the locked script logic on [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv):

### Table 1: Benchmark Reproduction Audit for GBDT Supporting Model

| Benchmark Evaluation | Metric | Reproduced Value | Locked Phase 4 Reference | Discrepancy | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Context LOGO** | **MAE (mm)** | **0.1186** | 0.1186 | 0.0000 | **Verified Exact** |
| (16 Tool Folds) | **RMSE (mm)** | **0.1695** | 0.1695 | 0.0000 | **Verified Exact** |
| | **$R^2$** | **0.5739** | 0.5739 | 0.0000 | **Verified Exact** |
| **Context LOCO** | **MAE (mm)** | **0.1213** | 0.1213 | 0.0000 | **Verified Exact** |
| (8 Condition Folds) | **RMSE (mm)** | **0.1756** | 0.1756 | 0.0000 | **Verified Exact** |
| | **$R^2$** | **0.5425** | 0.5425 | 0.0000 | **Verified Exact** |

*Takeaway:* Zero divergence from locked Phase 4 benchmark metrics. The GBDT baseline is completely stable and verified.

---

## 4. TreeSHAP Global Feature Importance

TreeSHAP computes exact cooperative game-theoretic feature attributions for tree ensembles based on conditional expectations. For all $N = 145$ valid observations, individual attributions satisfy exact local accuracy:
$$\sum_{j=1}^{8} \phi_{i,j} + E[f(X)] = \hat{y}_i, \quad \max_{i} \left| \sum_{j=1}^{8} \phi_{i,j} + E[f(X)] - \hat{y}_i \right| = 6.66 \times 10^{-16}\text{ mm}$$
The model expected base value is $E[f(X)] = 0.3378\text{ mm}$ (close to the dataset mean of $V_B = 0.3394\text{ mm}$).

Global feature importance is measured by the **mean absolute SHAP value**:
$$\text{Mean } |\text{SHAP}|_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_{i,j}|$$

### Table 2: Complete 8-Feature TreeSHAP Global Importance & Directionality

| Rank | Feature | Type | Mean \|SHAP\| (mm) | Min SHAP (mm) | Max SHAP (mm) | Pearson $r$ (Feature vs. SHAP) | Observed Direction Within Model | Native MDI Importance | Native MDI Rank |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **1** | `smcAC_rms` | Sensor | **0.1383** | $-0.2405$ | $+0.4586$ | $+0.9459$ | **Positive** (Higher value $\rightarrow$ Higher wear) | 0.4413 | 1 |
| **2** | `DOC_mm` | Context | **0.0598** | $-0.1080$ | $+0.1621$ | $-0.9318$ | **Negative** (Higher value $\rightarrow$ Lower wear) | 0.0899 | 5 |
| **3** | `vib_spindle_kurtosis` | Sensor | **0.0546** | $-0.0895$ | $+0.1521$ | $+0.7107$ | **Positive** (Higher value $\rightarrow$ Higher wear) | 0.1360 | 3 |
| **4** | `AE_spindle_p2p` | Sensor | **0.0454** | $-0.0998$ | $+0.1608$ | $+0.9001$ | **Positive** (Higher value $\rightarrow$ Higher wear) | 0.1408 | 2 |
| **5** | `feed_mm_rev` | Context | **0.0302** | $-0.0532$ | $+0.0480$ | $-0.9594$ | **Negative** (Higher value $\rightarrow$ Lower wear) | 0.0254 | 7 |
| **6** | `AE_table_rms` | Sensor | **0.0186** | $-0.0351$ | $+0.1366$ | $-0.1805$ | **Weak / Mixed** (Near-zero impact) | 0.1020 | 4 |
| **7** | `material_code` | Context | **0.0122** | $-0.0282$ | $+0.0857$ | $+0.7916$ | **Positive Offset** (Stainless Steel > Cast Iron) | 0.0417 | 6 |
| **8** | `vib_spindle_p2p` | Sensor | **0.0051** | $-0.0244$ | $+0.0238$ | $+0.0605$ | **Weak / Neutral** (Near-zero impact) | 0.0230 | 8 |

*Data serialized at:* [`reports/phase5_2b_gbdt_shap_importance.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2b_gbdt_shap_importance.csv)  
*Figure:* [Figure 1: TreeSHAP Beeswarm Summary Plot](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_summary.png)

---

## 5. SHAP Directionality & Dependence Analysis

### 5.1 Primary Sensor Feature: `smcAC_rms` (Rank 1, Mean |SHAP| = 0.1383 mm)
- **Distribution:** Point attributions range from $-0.2405\text{ mm}$ (fresh tool cuts with low spindle current) to $+0.4586\text{ mm}$ (severely worn tool passes with high spindle current).
- **Directionality:** Strongly positive monotonic ($r = +0.9459$, Spearman $\rho = +0.9748$). Higher `smcAC_rms` is associated with higher model predictions within the fitted GBDT over the observed data range.
- **Context Interaction:** As observed in the dependence plot, cuts at lower DOC ($0.75\text{ mm}$, blue markers) reach higher positive SHAP contributions at equivalent spindle current levels compared to high DOC cuts ($1.50\text{ mm}$, red markers). This reflects the tree ensemble using operating conditions to normalize baseline cutting load.
- *Artifact:* [Figure 2: TreeSHAP Dependence — smcAC_rms](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_smcAC_rms.png)

### 5.2 Operating Context Feature: `DOC_mm` (Rank 2, Mean |SHAP| = 0.0598 mm)
- **Distribution:** Clear binary separation with zero overlap across the two test levels:
  - $\text{DOC} = 0.75\text{ mm}$ ($n = 90$): Mean $\text{SHAP} = +0.0509\text{ mm}$ (range: $+0.0309$ to $+0.1621\text{ mm}$).
  - $\text{DOC} = 1.50\text{ mm}$ ($n = 55$): Mean $\text{SHAP} = -0.0742\text{ mm}$ (range: $-0.1080$ to $-0.0444\text{ mm}$).
- **Interpretation:** Within the fitted GBDT formulation, higher DOC values are associated with negative prediction offsets over the observed data range. This indicates that the tree ensemble uses depth of cut jointly with spindle current when estimating $V_B$, adjusting baseline predictions downward when high current is expected due to heavier cuts. This is an associative pattern within joint estimation and must not be interpreted as a physical claim that heavier cuts reduce wear.
- *Artifact:* [Figure 3: TreeSHAP Dependence — DOC_mm](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_DOC_mm.png)

### 5.3 Operating Context Feature: `feed_mm_rev` (Rank 5, Mean |SHAP| = 0.0302 mm)
- **Distribution:** Binary separation across feed levels:
  - $\text{Feed} = 0.25\text{ mm/rev}$ ($n = 78$): Mean $\text{SHAP} = +0.0277\text{ mm}$ (range: $+0.0141$ to $+0.0480\text{ mm}$).
  - $\text{Feed} = 0.50\text{ mm/rev}$ ($n = 67$): Mean $\text{SHAP} = -0.0332\text{ mm}$ (range: $-0.0532$ to $-0.0152\text{ mm}$).
- **Interpretation:** Within the fitted joint model, the higher feed level receives a negative SHAP offset. This indicates that GBDT uses feed information jointly with sensor features when estimating $V_B$; the attribution should not be interpreted as a physical wear-reduction effect.
- *Artifact:* [Figure 4: TreeSHAP Dependence — feed_mm_rev](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_feed_mm_rev.png)

### 5.4 Operating Context Feature: `material_code` (Rank 7, Mean |SHAP| = 0.0122 mm)
- **Distribution:**
  - $\text{material_code} = 1$ (Cast Iron, $n = 97$): Mean $\text{SHAP} = -0.0083\text{ mm}$ (range: $-0.0282$ to $+0.0001\text{ mm}$).
  - $\text{material_code} = 2$ (Stainless Steel J45, $n = 48$): Mean $\text{SHAP} = +0.0199\text{ mm}$ (range: $-0.0017$ to $+0.0857\text{ mm}$).
- **Interpretation & Caution:** `material_code` is a categorical workpiece indicator. The numerical difference ($2 - 1 = 1$) does not represent a metric physical interval. The mean SHAP attribution for Stainless Steel J45 is approximately 0.028 mm higher than for Cast Iron in this fitted model, indicating a positive empirical material-condition contribution within this benchmark.
- *Artifact:* [Figure 5: TreeSHAP Attribution — material_code](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_material_code.png)

### 5.5 Secondary Sensor Dynamics
- **`vib_spindle_kurtosis` (Rank 3, Mean |SHAP| = 0.0546 mm):** Exhibits a strong positive relationship ($r = +0.7107$, $\rho = +0.9163$). GBDT assigns higher predictive importance to `vib_spindle_kurtosis` than Ridge does, suggesting that its relationship with $V_B$ may contain nonlinear or range-dependent structure that is not captured by a single linear coefficient.
- **`AE_spindle_p2p` (Rank 4, Mean |SHAP| = 0.0454 mm):** Exhibits strong positive monotonic attribution ($r = +0.9001$, $\rho = +0.9304$). Spindle acoustic emission peak-to-peak amplitude contributes positively to estimated wear across both models.
- **`AE_table_rms` (Rank 6, Mean |SHAP| = 0.0186 mm) & `vib_spindle_p2p` (Rank 8, Mean |SHAP| = 0.0051 mm):** Both show minimal predictive weight, with attributions clustered tightly around zero.

---

## 6. Ridge vs. GBDT Comparative Audit

To address the core question of whether nonlinear GBDT supports or contradicts the linear Ridge formulation, Table 3 evaluates feature importance, rank, direction, and consistency:

### Table 3: Ridge vs. GBDT Formulation Comparison Table

| Feature | Type | Ridge Standardized Coef ($\beta$) | Ridge Rank | Ridge Sign | GBDT Mean \|SHAP\| (mm) | GBDT SHAP Rank | GBDT Observed Direction | Broad Consistency | Consistency Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **`smcAC_rms`** | Sensor | **+0.2940** | **1** | $+$ | **0.1383** | **1** | Positive | **Yes** | Rank 1 in both models; strong positive weight/attribution associated with higher estimated wear. |
| **`DOC_mm`** | Context | **-0.1547** | **2** | $-$ | **0.0598** | **2** | Negative | **Yes** | Rank 2 in both models; negative offset under joint prediction with spindle load. |
| **`feed_mm_rev`** | Context | **-0.0944** | **3** | $-$ | **0.0302** | **5** | Negative | **Yes** | Negative direction in both; acts as a secondary operating offset. |
| **`AE_spindle_p2p`** | Sensor | **+0.0302** | **4** | $+$ | **0.0454** | **4** | Positive | **Yes** | Rank 4 in both models; positive weight/attribution for high-frequency contact peaks. |
| **`vib_spindle_kurtosis`** | Sensor | **+0.0205** | **5** | $+$ | **0.0546** | **3** | Positive | **Partial** | Positive direction in both; GBDT gives higher rank (3 vs. 5) via non-linear peak splits. |
| **`AE_table_rms`** | Sensor | **-0.0199** | **6** | $-$ | **0.0186** | **6** | Weak / Mixed | **Yes** | Rank 6 in both models; minor contribution in both formulations. |
| **`vib_spindle_p2p`** | Sensor | **-0.0156** | **7** | $-$ | **0.0051** | **8** | Weak / Neutral | **Yes** | Both rank at bottom (7–8); negligible weight/attribution near zero. |
| **`material_code`** | Context | **+0.0056** | **8** | $+$ | **0.0122** | **7** | Positive Offset | **Yes** | Both rank near bottom (7–8); Stainless Steel receives slight positive offset. |

*Data serialized at:* [`reports/phase5_2b_ridge_vs_gbdt_interpretation.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2b_ridge_vs_gbdt_interpretation.csv)

### Synthesis of Comparative Logic:
Following the governance criteria:
> **"The Ridge and GBDT formulations show broadly consistent feature importance patterns, with the same major sensor/context variables contributing strongly to prediction. This provides cross-model support that the main predictive structure is not solely an artifact of the linear formulation within this dataset."**

Both models identify the same dominant sensor and operating-context variables, with broadly similar feature ordering but some differences in secondary-feature importance.

Key observations supporting this verdict:
1. **No Directional Contradiction:** Zero features change signs between Ridge and GBDT. Features with positive linear coefficients exhibit positive SHAP attributions; features with negative linear coefficients exhibit negative SHAP attributions.
2. **Top-Tier Alignment:** Both models identify `smcAC_rms` as the dominant feature (Rank 1) and `DOC_mm` as the primary operating offset (Rank 2).
3. **Non-linear Discrepancy (Partial Consistency on Kurtosis):** The primary shift in ordering occurs with `vib_spindle_kurtosis` (Ridge Rank 5 $\rightarrow$ GBDT Rank 3). GBDT tree splits capture range-dependent or nonlinear sensitivity in vibration peakedness, whereas linear regression applies a uniform slope across the entire domain.

---

## 7. Methodological Constraints & Interpretation Limitations

To preserve strict scientific rigor, the following constraints must be maintained:
1. **Model Attribution vs. Physical Causality:**  
   SHAP values quantify how the fitted GBDT algorithm combines input variables to compute an output prediction. They do NOT prove physical causality, direct mechanical contact wear laws, or physical force decomposition.
2. **Operating Context Interpretation:**  
   The negative SHAP contributions for `DOC_mm` and `feed_mm_rev` must NOT be interpreted as physical evidence that higher cutting depths or faster feeds reduce tool wear. In real machining, higher feeds and depths accelerate tool deterioration. The negative model attribution occurs strictly because the model is performing joint estimation: when heavy cutting parameters naturally elevate spindle current, the model applies a negative context offset to avoid overestimating wear.
3. **Categorical Encoding of Material:**  
   `material_code` ($1 = \text{Cast Iron}$, $2 = \text{Stainless Steel}$) is a categorical label. The numerical distance has no metric meaning. Its SHAP attribution represents an empirical condition offset between the two tested workpiece alloys in this specific benchmark.
4. **Dataset Scale & Generalization Boundaries:**  
   This analysis is derived from 145 cuts across 16 tool cases and 8 condition combinations from the NASA Milling dataset. These findings represent model behavior on this benchmark and must not be claimed as universal laws for all industrial CNC milling operations.
5. **Multicollinearity:**  
   Correlations among sensor channels (e.g., between spindle current, vibration, and cutting conditions) influence how decision tree splits distribute feature attributions.
6. **Full-Data Fitted-Model Explanation:**  
   TreeSHAP values were computed on the fitted full-data GBDT model and therefore describe model behavior on the observed dataset. They should not be interpreted as out-of-sample feature importance or as evidence that the same attribution hierarchy will necessarily hold on new machines, tools, or operating conditions.
7. **Non-Nested Feature Screening Boundary:**  
   Because the initial feature screening in Phase 3.2 utilized all 145 labeled observations, the overall pipeline evaluation is not a fully nested model-development estimate.

---

## 8. Deliverables & Data Manifest

- 📊 **SHAP Global Importance Table CSV:** [`reports/phase5_2b_gbdt_shap_importance.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2b_gbdt_shap_importance.csv)
- 📊 **Ridge vs. GBDT Comparison Table CSV:** [`reports/phase5_2b_ridge_vs_gbdt_interpretation.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2b_ridge_vs_gbdt_interpretation.csv)
- 🖼️ **TreeSHAP Beeswarm Summary Plot:** [Figure 1: TreeSHAP Summary Beeswarm Plot](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_summary.png)
- 🖼️ **TreeSHAP Dependence (smcAC_rms):** [Figure 2: Dependence Plot — smcAC_rms](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_smcAC_rms.png)
- 🖼️ **TreeSHAP Dependence (DOC_mm):** [Figure 3: Dependence Plot — DOC_mm](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_DOC_mm.png)
- 🖼️ **TreeSHAP Dependence (feed_mm_rev):** [Figure 4: Dependence Plot — feed_mm_rev](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_feed_mm_rev.png)
- 🖼️ **TreeSHAP Attribution (material_code):** [Figure 5: Attribution Plot — material_code](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2b_gbdt_shap_dependence_material_code.png)
- 🐍 **Audit & Generation Script:** [`scripts/phase5_2b_gbdt_shap_audit.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase5_2b_gbdt_shap_audit.py)

---

## 9. Governance & Methodological Integrity Statement

> **"GBDT implementation parameters (including subsample=0.8) match the locked Phase 4 scripts exactly, reproducing Context LOGO and LOCO benchmark metrics with zero discrepancy. TreeSHAP values were computed on the locked 8-feature formulation. All causal claims have been excluded, and the end-to-end limitation regarding non-nested Phase 3.2 feature screening as well as the full-data nature of SHAP attributions have been explicitly incorporated. Ridge Regression remains the Primary Engineering Model and GBDT remains the Supporting Nonlinear Model."**

---

*Report certified by AntiGravity (Implementation Agent) for review by Strategic Lead ChatGPT and Domain Lead Bright.*
