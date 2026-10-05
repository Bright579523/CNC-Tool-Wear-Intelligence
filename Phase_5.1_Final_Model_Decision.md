# Phase 5.1 — Final Model Decision Review
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2) — CNC Machining & Tool Wear Analytics
- **Document Type:** Strategic Final Model Selection & Architecture Review
- **Author:** AntiGravity (Implementation Agent)
- **Reviewers:** Bright (Domain Lead) & ChatGPT (Strategic Lead)
- **Status:** Phase 5.1 Complete (Final Correction Pass Applied) — Source-of-Truth Decision Locked for Phase 5

---

## 1. Objective

Phase 1 through Phase 4 established the foundational data pipeline, manufacturing engineering analysis, compact sensor feature engineering, and rigorous cross-validation stress tests (LOGO and LOCO).

The objective of **Phase 5.1** is to evaluate the existing evidence generated across Phase 4 and formulate a disciplined, evidence-based decision regarding which machine learning model(s) should be carried forward into Phase 5 (Final Model Formulation, Feature Interpretation via SHAP, and Practical Guidelines).

In accordance with strict project governance:
- **Zero new experiments or retraining:** No models are retrained, tuned, or modified during Phase 5.1.
- **Multi-criteria evaluation:** Candidate selection is not based on a single metric (such as maximum $R^2$), but on a balanced synthesis of:
  1. Generalization across unseen tools and unseen combinations of known operating factors.
  2. Prediction error magnitude (prioritizing MAE in millimeters of flank wear).
  3. Error stability across wear lifecycle bins (low, moderate, and high wear).
  4. Model interpretability for an engineering audience.
  5. Model complexity relative to the small-sample manufacturing dataset (145 observations across 16 tools and 8 cutting conditions).

The review concludes with explicit designations for:
- **Primary Final Model**
- **Supporting / Comparison Model**
- **Models Not Carried Forward**

---

## 2. Locked Methodology

All evaluations reviewed herein strictly adhere to the methodology established and frozen in Phase 3.2 and Phase 4:

### 2.1 Predictive Target
- Continuous flank wear: `VB_mm` (observed range: $0.0000$ to $1.5300\text{ mm}$, mean $0.3394\text{ mm}$, std $0.2595\text{ mm}$).

### 2.2 Input Feature Spaces
- **Primary Sensor Features (5 features):**
  1. `smcAC_rms` (Spindle motor AC current root mean square)
  2. `vib_spindle_kurtosis` (Spindle vibration signal peakedness)
  3. `vib_spindle_p2p` (Spindle vibration peak-to-peak amplitude)
  4. `AE_table_rms` (Table acoustic emission root mean square)
  5. `AE_spindle_p2p` (Spindle acoustic emission peak-to-peak amplitude)
- **Operating Context Features (3 features):**
  1. `material_code` (Categorical binary: Cast Iron = 0, Stainless Steel J45 = 1)
  2. `DOC_mm` (Depth of cut, $a_p \in \{0.75, 1.50\text{ mm}\}$)
  3. `feed_mm_rev` (Feed rate per revolution, $f \in \{0.25, 0.50\text{ mm/rev}\}$)
- **Forbidden Inputs:** Identifiers (`case`, `run`, `condition_id`) and temporal proxies (`cumulative_time_min`) remain strictly excluded from model feature spaces.

### 2.3 Cross-Validation Frameworks
- **LOGO (Leave-One-Group-Out / Leave-One-Case-Out):** Exactly 16 outer folds. In each fold, all cuts from one physical tool case are held out. The training set retains the other tool case operating under the identical cutting condition.
- **LOCO (Leave-One-Condition-Out):** Exactly 8 outer folds. In each fold, both tool cases operating under the held-out cutting condition are isolated exclusively in the test set. LOCO evaluates **generalization to previously unseen combinations of known operating factors** (material, DOC, and feed), not extrapolation to unencountered parameter values.

### 2.4 Dataset Scale
- Exactly **145 valid cuts** across **16 tool cases**, grouped into **8 distinct cutting-condition combinations** (2 tools per condition).

---

## 3. Candidate Models

Four machine learning architectures with distinct structural inductive biases were evaluated across Phase 4:

1. **Ridge Regression:**
   - Linear regression regularized by an $L_2$ penalty ($\alpha = 1.0$).
   - Closed-form analytical solution; assumes linear relationships in standardized feature space; highly transparent coefficients.
2. **Support Vector Regression (SVR):**
   - Non-linear kernel regression using Radial Basis Function (RBF) kernel ($C = 1.0$, $\epsilon = 0.05$, $\gamma = \text{'scale'}$).
   - Maps inputs into a high-dimensional feature space; creates a margin-based $\epsilon$-insensitive loss tube.
3. **Random Forest (RF):**
   - Non-linear ensemble of 100 randomized bagged decision trees (`max_depth=5`, `random_state=42`).
   - Non-parametric step-function predictions; reduces variance via feature and sample bagging.
4. **Gradient Boosting Decision Trees (GBDT):**
   - Non-linear sequential boosting ensemble of 100 shallow decision trees (`max_depth=3`, `learning_rate=0.05`, `random_state=42`).
   - Fits consecutive trees to pseudo-residuals; captures feature interactions with restricted tree depth.

---

## 4. Performance Comparison

The quantitative results below are retrieved directly from the locked Phase 4 artifacts:
- [`reports/phase4_1_3_ml_benchmark.md`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_1_3_ml_benchmark.md)
- [`reports/phase4_5_context_fusion/context_fusion_summary.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_5_context_fusion/context_fusion_summary.csv)
- [`reports/phase4_5B_LOCO/loco_summary.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_5B_LOCO/loco_summary.csv)
- [`reports/phase4_5B_LOCO/logo_vs_loco_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase4_5B_LOCO/logo_vs_loco_comparison.csv)

### 4.1 Sensor-Only LOGO Benchmark (Phase 4.1–4.3)
Evaluated across 16 folds using only the 5 Primary Sensor Features:

| Model | MAE (mm) | RMSE (mm) | $R^2$ | vs. Baseline MAE Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Dummy Baseline** | 0.1998 | 0.2644 | -0.0367 | Baseline |
| **Ridge Regression** | **0.1361** | 0.1988 | 0.4139 | +31.86% |
| **Gradient Boosting** | **0.1381** | **0.1972** | **0.4229** | +30.88% |
| **Random Forest** | 0.1412 | 0.2077 | 0.3604 | +29.33% |
| **Support Vector Regression** | 0.1481 | 0.2107 | 0.3415 | +25.88% |

*Takeaway:* Sensor signals alone contain measurable wear information, lowering MAE by $\approx 26\%\text{--}32\%$ relative to the naive mean baseline. Ridge and Gradient Boosting performed best among the sensor-only candidates.

---

### 4.2 Context-Fused LOGO Benchmark (Phase 4.5A)
Evaluated across 16 folds fusing the 5 Sensor Features with the 3 Operating Context Features (8 features total):

| Model | Feature Set | LOGO MAE (mm) | LOGO RMSE (mm) | LOGO $R^2$ | $\Delta \text{MAE}$ vs Sensor (%) | Folds Improved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVR** | Primary 5 + Context (8 feats) | **0.1018** | 0.1532 | 0.6518 | **-31.28%** | 14 / 16 (87.5%) |
| **Ridge** | Primary 5 + Context (8 feats) | **0.1044** | **0.1425** | **0.6989** | **-23.29%** | 12 / 16 (75.0%) |
| **Gradient Boosting** | Primary 5 + Context (8 feats) | 0.1186 | 0.1695 | 0.5739 | -14.12% | 14 / 16 (87.5%) |
| **Random Forest** | Primary 5 + Context (8 feats) | 0.1297 | 0.1934 | 0.4452 | -8.16% | 13 / 16 (81.2%) |

*Takeaway:* Context Fusion substantially reduced prediction error across all models. SVR and Ridge achieved the lowest MAE ($0.1018\text{--}0.1044\text{ mm}$), while Ridge yielded the highest $R^2$ ($0.6989$) and lowest RMSE ($0.1425\text{ mm}$).

---

### 4.3 Context-Fused LOCO Generalization Benchmark (Phase 4.5B)
Evaluated across 8 outer condition folds (holding out entire cutting-condition combinations):

| Model | Sensor LOCO MAE (mm) | Context LOCO MAE (mm) | Context LOCO RMSE (mm) | Context LOCO $R^2$ | $\Delta \text{MAE}$ with Context (%) | Conditions Improved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVR** | 0.1528 | **0.1149** | 0.1655 | 0.5939 | **-24.78%** | 6 / 8 (75.0%) |
| **Gradient Boosting** | 0.1459 | **0.1213** | 0.1756 | 0.5425 | -16.84% | **8 / 8 (100.0%)** |
| **Ridge** | 0.1468 | **0.1221** | **0.1566** | **0.6360** | -16.80% | 5 / 8 (62.5%) |
| **Random Forest** | 0.1513 | 0.1433 | 0.2037 | 0.3848 | -5.31% | 6 / 8 (75.0%) |

*Takeaway:* Context Fusion maintained superiority over sensor-only models even under LOCO. SVR attained the lowest LOCO MAE ($0.1149\text{ mm}$). Gradient Boosting ($0.1213\text{ mm}$) and Ridge ($0.1221\text{ mm}$) were practically similar ($\Delta \text{MAE} = 0.0008\text{ mm}$). Gradient Boosting achieved improvement in all 8 held-out conditions. Random Forest lagged behind all other candidates.

---

## 5. Generalization Assessment

Evaluating generalization is the highest-priority criterion. The transition from LOGO (unseen tool under known conditions) to LOCO (unseen tool under an unseen operating combination) quantifies the **Generalization Gap**:

### Table 5: LOGO vs. LOCO Generalization Gap (Context Fusion Models)

| Model | LOGO MAE (mm) | LOCO MAE (mm) | Absolute Gap (mm) | Relative Gap (%) | LOGO $R^2$ | LOCO $R^2$ | Conditions Improved under LOCO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 0.1186 | 0.1213 | **+0.0027** | **+2.30%** | 0.5739 | 0.5425 | **8 / 8 (100.0%)** |
| **Random Forest** | 0.1297 | 0.1433 | +0.0136 | +10.47% | 0.4452 | 0.3848 | 6 / 8 (75.0%) |
| **SVR** | 0.1018 | 0.1149 | +0.0131 | +12.90% | 0.6518 | 0.5939 | 6 / 8 (75.0%) |
| **Ridge** | 0.1044 | 0.1221 | +0.0177 | +16.94% | 0.6989 | 0.6360 | 5 / 8 (62.5%) |

### Generalization Synthesis:
1. **Gradient Boosting exhibited the highest condition consistency:**  
   GBDT demonstrated a minimal generalization gap ($+0.0027\text{ mm}$, $+2.30\%$) and was the only model to improve with context features in **100% (8/8) of held-out condition folds**. It handled held-out condition combinations without substantial performance degradation.
2. **Ridge demonstrated moderate generalization stability:**  
   Ridge's generalization gap ($+0.0177\text{ mm}$, $+16.94\%$) reflects the constraint of a single global linear hyper-plane when operating conditions shift. Ridge showed higher error on C1–C3 under LOCO, indicating that its linear formulation did not transfer equally well across all held-out conditions. Despite this gap, Ridge's LOCO MAE ($0.1221\text{ mm}$) remained practically similar to GBDT ($0.1213\text{ mm}$) and retained the highest LOCO $R^2$ ($0.6360$).
3. **SVR generalized effectively in aggregate, but with higher variance:**  
   SVR showed a moderate gap ($+0.0131\text{ mm}$, $+12.90\%$) and achieved the lowest absolute LOCO MAE ($0.1149\text{ mm}$). However, it deteriorated on Condition C8 (where error increased with context) and showed wider error dispersion on Stainless Steel.
4. **Random Forest exhibited poor generalization:**  
   Random Forest showed the highest absolute error across both settings ($\text{MAE} = 0.1433\text{ mm}$) and the lowest explained variance ($R^2 = 0.3848$).

---

## 6. Error Behavior Assessment

The detailed residual audit conducted in Phase 4.6 provides critical qualitative criteria:

### Table 6: Model Performance and Bias Across Analytical Wear Bins (Phase 4.6)

| Wear Bin | Actual Mean $V_B$ | Metric | Ridge | SVR | Random Forest | Gradient Boosting |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Low Wear** | 0.102 mm | **MAE (mm)** | 0.0940 | **0.0737** | 0.0854 | **0.0769** |
| ($V_B < 0.20\text{ mm}$, $n=49$) | | **Mean Bias (mm)** | +0.0060 | +0.0602 | +0.0581 | +0.0430 |
| | | **Predicted Range (mm)** | [-0.18, 0.30] | [0.09, 0.45] | [0.05, 0.61] | [0.03, 0.54] |
| **Moderate Wear** | 0.295 mm | **MAE (mm)** | **0.0761** | **0.0888** | 0.1099 | 0.1035 |
| ($0.20 \le V_B \le 0.40\text{ mm}$, $n=48$) | | **RMSE (mm)** | **0.0969** | 0.1167 | 0.1278 | 0.1290 |
| *(Study Threshold Window)* | | **Mean Bias (mm)** | +0.0305 | +0.0485 | +0.0485 | +0.0466 |
| **High Wear** | 0.626 mm | **MAE (mm)** | **0.1433** | **0.1436** | 0.1948 | 0.1762 |
| ($V_B > 0.40\text{ mm}$, $n=48$) | | **Mean Bias (mm)** | -0.0495 | -0.0864 | -0.1270 | -0.1014 |
| | | **Max Predicted (mm)** | 0.96 mm | 0.90 mm | 0.77 mm | 0.98 mm |

### Error Diagnostic Observations:
1. **Moderate Wear Precision (Ridge Advantage):**  
   Around the study's analytical tool-life threshold ($V_B = 0.30\text{ mm}$), Ridge achieved the lowest error of any candidate ($\text{MAE} = \mathbf{0.0761\text{ mm}}$, $\text{RMSE} = 0.0969\text{ mm}$), outperforming SVR ($0.0888\text{ mm}$) and GBDT ($0.1035\text{ mm}$).
2. **Low-Wear Behavior & Extrapolation Limits:**  
   Non-linear models (SVR, RF, GBDT) displayed positive bias ($+0.043\text{ to } +0.060\text{ mm}$) at low wear, consistent with a sensor baseline floor effect. Conversely, Ridge had near-zero mean bias ($+0.0060\text{ mm}$) because it extrapolated downward into negative predictions (minimum predicted wear = $-0.18\text{ mm}$). Non-negative output clipping may be considered as an engineering implementation constraint in deployment, though benchmark metrics reflect raw linear predictions.
3. **High-Wear Range Compression:**  
   All models exhibited prediction compression toward the central wear range, systematically underpredicting high wear ($V_B > 0.40\text{ mm}$). Random Forest was the most restricted (capping at $0.77\text{ mm}$), whereas GBDT and Ridge reached $0.96\text{--}0.98\text{ mm}$.
4. **Sensitivity to Outlier Case 13:**  
   In Stainless Steel Case 13 (where wear escalated to $1.53\text{ mm}$), Ridge had the lowest MAE ($0.2348\text{ mm}$), followed by SVR ($0.2598\text{ mm}$), GBDT ($0.3112\text{ mm}$), and RF ($0.3708\text{ mm}$). When excluding Case 13, Ridge achieved an MAE of **$0.1076\text{ mm}$** across the remaining 7 Stainless tools.

---

## 7. Interpretability and Complexity

The dataset consists of **145 samples across 16 tools**. In small manufacturing datasets, model architecture complexity must be rigorously weighed against interpretability and risk of spurious fitting:

### Table 7: Structural Comparison of Candidate Architectures

| Dimension | Ridge Regression | Support Vector Regression | Random Forest | Gradient Tree Boosting |
| :--- | :--- | :--- | :--- | :--- |
| **Model Family** | Regularized Linear ($L_2$) | Kernel Method (RBF) | Bagging Ensemble | Boosting Ensemble |
| **Model Complexity** | Very Low (9 parameters) | Moderate (Kernel SVs) | Higher than Ridge (100 depth-limited trees) | Moderate (100 depth-3 trees) |
| **Overfitting Risk on Small Data** | Very Low (Constrained) | Moderate (Sensitive to $C, \gamma$) | Moderate-High | Low-Moderate (Restricted depth) |
| **Engineering Interpretability** | **High** (Standardized $\beta$ weights) | **Low** (Dual-space kernel weights) | **Moderate** (Gini / Impurity) | **High via TreeSHAP** |
| **Physical Directionality** | Direct (Sign & magnitude of $\beta$) | Non-intuitive | Non-directional | Non-linear response curves |

---

## 8. Model-by-Model Assessment

### 8.1 Ridge Regression
1. **Performance Evidence:**  
   Demonstrated strong performance across all phases. Highest $R^2$ in both Context LOGO ($0.6989$) and Context LOCO ($0.6360$). Lowest RMSE in both settings ($0.1425\text{ mm}$ LOGO, $0.1566\text{ mm}$ LOCO). Lowest MAE in the moderate wear range ($0.0761\text{ mm}$).
2. **Behavior under LOGO:**  
   $\text{MAE} = 0.1044\text{ mm}$, improved in 12/16 folds ($-23.29\%$ error reduction with context).
3. **Behavior under LOCO:**  
   $\text{MAE} = 0.1221\text{ mm}$, improved in 5/8 conditions. Showed higher error on C1–C3 under LOCO, indicating that its linear formulation did not transfer equally well across all held-out conditions.
4. **Main Strengths:**  
   Maximum structural transparency; coefficients directly convey standardized relative weights to manufacturing engineers; lowest error near the study's analytical threshold; minimal parameter count.
5. **Main Weaknesses:**  
   Cannot model non-linear sensor-condition interactions without manual interaction terms; extrapolates into negative wear values at low wear unless clipped at zero; larger generalization gap ($+16.9\%$) between LOGO and LOCO than GBDT.
6. **Verdict as Final Primary Model:** **Highly Suitable.**  
   Its combination of low error near the analytical threshold, high linear transparency, and minimal parameter count makes it an exemplary primary model for this small-sample dataset.
7. **Verdict as Supporting Model:** Suitable as a transparent linear baseline if another model were selected as primary.

---

### 8.2 Support Vector Regression (SVR)
1. **Performance Evidence:**  
   Achieved the lowest overall MAE in both Context LOGO ($0.1018\text{ mm}$) and Context LOCO ($0.1149\text{ mm}$).
2. **Behavior under LOGO:**  
   $\text{MAE} = 0.1018\text{ mm}$, improved in 14/16 folds ($-31.28\%$ error reduction with context).
3. **Behavior under LOCO:**  
   $\text{MAE} = 0.1149\text{ mm}$, improved in 6/8 conditions.
4. **Main Strengths:**  
   Effective non-linear boundary estimation; margin-based loss provides robustness against small residual fluctuations.
5. **Main Weaknesses:**  
   Black-box architecture; dual-space support vector representations cannot be directly interpreted by manufacturing practitioners; hyperparameter tuning on small datasets risks instability; showed condition-level deterioration on C8.
6. **Verdict as Final Primary Model:** **Not Recommended.**  
   While SVR achieved the lowest numerical LOCO MAE ($0.1149\text{ mm}$), this slight predictive advantage is insufficient when considered alongside interpretability, condition consistency, and engineering usability.
7. **Verdict as Supporting Model:** Subordinate to GBDT, which provides superior condition-level consistency (8/8) and clearer tree-based interpretability.

---

### 8.3 Random Forest (RF)
1. **Performance Evidence:**  
   Consistently ranked lowest among the four ML models across all Phase 4 benchmarks. Context LOGO $\text{MAE} = 0.1297\text{ mm}$ ($R^2 = 0.4452$); Context LOCO $\text{MAE} = 0.1433\text{ mm}$ ($R^2 = 0.3848$).
2. **Behavior under LOGO:**  
   Modest gain with context features ($-8.16\%$), lagging other models by $0.011\text{--}0.028\text{ mm}$ in MAE.
3. **Behavior under LOCO:**  
   Smallest context benefit ($-5.31\%$), failing to match linear or boosting performance.
4. **Main Strengths:**  
   Non-parametric flexibility; resistant to gross outliers.
5. **Main Weaknesses:**  
   Severe prediction compression; worst severe-wear saturation (predictions capped at $0.77\text{ mm}$); highest residual slope ($-0.5225$); utilizes 100 depth-limited trees (`max_depth=5`), representing higher structural complexity than Ridge, with limited empirical benefit in this dataset.
6. **Verdict as Final Primary Model:** **Unsuitable.**
7. **Verdict as Supporting Model:** **Not Recommended.** Does not provide compelling engineering or predictive value over GBDT.

---

### 8.4 Gradient Boosting Decision Trees (GBDT)
1. **Performance Evidence:**  
   Strong across all evaluations. Context LOGO $\text{MAE} = 0.1186\text{ mm}$ ($R^2 = 0.5739$); Context LOCO $\text{MAE} = 0.1213\text{ mm}$ ($R^2 = 0.5425$).
2. **Behavior under LOGO:**  
   Improved in 14/16 folds ($-14.12\%$ error reduction with context).
3. **Behavior under LOCO:**  
   **Improved in 8/8 held-out conditions (100.0%)**; negligible generalization gap between LOGO and LOCO ($+0.0027\text{ mm}$, $+2.30\%$).
4. **Main Strengths:**  
   Superior generalization consistency across condition splits; restricted tree depth (`max_depth=3`) prevents rampant overfitting; captures non-linear interactions without manual feature engineering; in the observed cross-validation predictions, GBDT remained non-negative and reached a higher maximum predicted wear than Random Forest.
5. **Main Weaknesses:**  
   Higher MAE in the moderate wear range ($0.1035\text{ mm}$) compared to Ridge ($0.0761\text{ mm}$); higher complexity than a linear model; requires TreeSHAP for interpretation.
6. **Verdict as Final Primary Model:** **Highly Suitable.**  
   Demonstrated the highest condition-level generalization stability in the entire benchmark suite.
7. **Verdict as Supporting Model:** **Exemplary.**  
   Serves as the ideal non-linear companion model to validate and cross-examine linear predictions.

---

## 9. Final Model Decision

Based on the synthesis of predictive error, generalization stability, error behavior, and engineering interpretability:

```
+---------------------------------------------------------------------------------------+
|                               FINAL MODEL DECISION SUMMARY                            |
+---------------------------------------------------------------------------------------+
|  PRIMARY FINAL MODEL:                                                                 |
|  --> Ridge Regression (Context Fusion, 8 Features, α=1.0)                            |
|                                                                                       |
|  SUPPORTING / COMPARISON MODEL:                                                       |
|  --> Gradient Tree Boosting (Context Fusion, 8 Features, max_depth=3)                |
|                                                                                       |
|  MODELS NOT CARRIED FORWARD:                                                          |
|  --> Support Vector Regression (SVR)  [Archived as diagnostic reference]              |
|  --> Random Forest (RF)               [Dropped due to inferior accuracy & saturation] |
+---------------------------------------------------------------------------------------+
```

### 9.1 Primary Model: Ridge Regression (Context Fusion)
- **Role:** Main final predictive formulation for reporting, coefficient analysis, and baseline monitoring.
- **Specification:** Ridge Regression, context fusion (5 Primary Sensor Features + 3 Operating Context Features), $\alpha = 1.0$. Non-negative output clipping may be applied as an implementation constraint and is not part of the Phase 4 benchmark metrics.

### 9.2 Supporting Model: Gradient Tree Boosting (Context Fusion)
- **Role:** Non-linear reference and verification model carried into Phase 5 for TreeSHAP interaction analysis.
- **Specification:** 100 estimators, `learning_rate=0.05`, `max_depth=3`, 8 features.

### 9.3 Models Not Carried Forward
- **Random Forest:** Dropped entirely. Consistently produced the highest MAE, lowest $R^2$, and worst range compression across both LOGO and LOCO.
- **Support Vector Regression (SVR):** Dropped from the main presentation. While SVR achieved the lowest LOCO MAE ($0.1149\text{ mm}$), this slight predictive advantage is insufficient when considered alongside interpretability, condition consistency (improved in 6/8 conditions vs. GBDT's 8/8), and engineering usability.

---

## 10. Rationale

The decision to adopt a **Dual-Model Strategy** (Ridge Primary + GBDT Supporting) rests on the following evidence:

### 10.1 LOCO Parity Between Ridge and GBDT
Under the strictest generalization test (LOCO), Ridge and GBDT are practically similar in aggregate predictive error on this dataset:
- GBDT LOCO MAE: **$0.1213\text{ mm}$**
- Ridge LOCO MAE: **$0.1221\text{ mm}$**
- Difference: **$0.0008\text{ mm}$**

Because their predictive error on unseen condition combinations is practically similar, the selection between them must be guided by **interpretability, complexity, and specific regime strengths**.

### 10.2 Why Ridge as Primary?
1. **Superior Accuracy in the Operational Decision Window:**  
   In the Moderate Wear range ($0.20 \le V_B \le 0.40\text{ mm}$), surrounding the study's $0.30\text{ mm}$ analytical threshold, Ridge achieved $\text{MAE} = \mathbf{0.0761\text{ mm}}$, compared to GBDT's $0.1035\text{ mm}$ (a **$26.5\%$ precision advantage** where wear monitoring is most relevant).
2. **Maximum Engineering Interpretability:**  
   Ridge coefficients provide immediate, sign-and-magnitude transparency for manufacturing engineers. Each standardized coefficient expresses a clear, linear weighting of sensor signals and cutting parameters without black-box opacity.
3. **Parsimony and Sample Scale:**  
   For a dataset of 145 cuts across 16 tools, a regularized 9-parameter linear model represents an appropriately restrained model capacity, minimizing the risk of learning idiosyncratic noise artifacts.

### 10.3 Why GBDT as Supporting Model?
1. **High Condition-Level Consistency:**  
   GBDT improved with context in **8 out of 8 held-out conditions (100%)** and exhibited a minimal LOGO-to-LOCO generalization gap ($+2.3\%$). The result shows that the benefit of Context Fusion is not limited to the linear Ridge formulation, confirming that GBDT reproduced the context-fusion benefit under a non-linear model family.
2. **Non-Linear Verification via TreeSHAP:**  
   Carrying GBDT into Phase 5 enables TreeSHAP analysis to investigate non-linear feature interactions (e.g., how sensor current sensitivity changes between Cast Iron and Stainless Steel) that Ridge's linear structure cannot articulate.
3. **Non-Negative Predictions in Cross-Validation:**  
   In the observed cross-validation predictions, GBDT predictions remained non-negative across all cuts without post-hoc clipping.

---

## 11. Known Limitations

In accordance with disciplined scientific reporting, the final model architecture operates under the following documented constraints:

1. **Small-Sample Boundary:**  
   The dataset comprises only 16 tool cases across 8 cutting conditions. The reported generalization performance reflects cross-validation stress tests within this specific experimental envelope and should not be construed as proof of universal factory-floor readiness.
2. **High-Wear Prediction Compression:**  
   Both models show limited extrapolation at severe wear ($V_B > 0.40\text{ mm}$), capping predictions below $1.0\text{ mm}$ when true wear escalates past $1.2\text{ mm}$ (as in Case 13). The models function effectively for **monitoring wear progression up to the analytical threshold**, but cannot accurately quantify the magnitude of runaway catastrophic degradation.
3. **Linear Lower-Bound Extrapolation in Ridge:**  
   Unconstrained Ridge predictions can yield negative values on fresh tools (minimum observed $-0.18\text{ mm}$). Non-negative output clipping may be applied as an implementation constraint; it is not part of the Phase 4 benchmark metrics and will be evaluated in Phase 5.2/implementation.
4. **Decision Scope:**  
   The models are evaluated as **wear estimation / monitoring support tools**. They have not been evaluated as an autonomous replacement-decision system (which would require classification threshold analysis, false-replacement cost modeling, and operational risk metrics).

---

## 12. Phase 5.1 Conclusion

The Phase 4 empirical evidence establishes a clear, defensible path forward:
- **Ridge Regression** is selected as the **Primary Final Model**, offering the lowest prediction error in the critical moderate wear regime, high $R^2$ and RMSE performance, and maximum engineering transparency for a small manufacturing dataset.
- **Gradient Tree Boosting** is selected as the **Supporting Final Model**, providing non-linear verification, consistent condition-level improvement across all 8 held-out LOCO folds, and an ideal platform for Phase 5 TreeSHAP interaction analysis.
- **Random Forest** and **Support Vector Regression** are removed from the main final formulation.

**Phase 5.1 is complete.** The project is positioned to proceed directly to **Phase 5.2 (Final Model Formulation, Feature Importance & SHAP Interpretation)** using this locked two-model framework.

---

*Report certified by AntiGravity (Implementation Agent) under the guidance of Domain Lead Bright and Strategic Lead ChatGPT.*
