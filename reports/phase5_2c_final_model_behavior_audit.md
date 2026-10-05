# Phase 5.2C: Final Model Behavior & Robustness Audit
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2) — CNC Machining & Tool Wear Analytics
- **Document Type:** Final Model Behavior, Error Regime & Cross-Model Robustness Audit
- **Author:** AntiGravity (Implementation Agent)
- **Reviewers:** Bright (Domain Lead) & ChatGPT (Strategic Lead)
- **Status:** Phase 5.2C Complete & Locked — Final Modeling Chapter Concluded

---

## 1. Executive Summary & Objective

Phase 5.2C represents the **final behavioral audit of the locked Ridge Primary Model and GBDT Supporting Model** prior to closing Phase 5. The objective is strictly diagnostic and observational:
> **"Where do the two locked models perform well, where do they struggle, and how similar or different are their predictions?"**

This audit consolidates empirical evidence across all 145 labeled observations under the locked Leave-One-Tool-Out (LOGO, 16 folds) framework.

### Summary of Core Audit Findings:
1. **Strong Global Model Agreement:**  
   Ridge and GBDT predictions exhibit high linear concordance across the dataset ($\text{Pearson } r = 0.8451$, Mean Absolute Difference $\text{MAD} = 0.0987\text{ mm}$, and an overall mean signed difference of $-0.0007\text{ mm}$). GBDT largely reproduces the same predictive structure as Ridge rather than providing a conflicting prediction landscape.
2. **Wear Lifecycle Decomposition & Best-Performing Wear Regime:**  
   - **Moderate Wear Range ($0.20 \le V_B \le 0.40\text{ mm}$, $n = 48$):** Ridge performs at its peak in this window ($\text{MAE} = 0.0761\text{ mm}$, $\text{RMSE} = 0.0969\text{ mm}$), outperforming GBDT ($\text{MAE} = 0.1035\text{ mm}$) by $26.5\%$. This range directly surrounds the study's analytical tool-life threshold ($V_B = 0.30\text{ mm}$).
   - **Low Wear Range ($V_B < 0.20\text{ mm}$, $n = 49$):** In the low-wear regime, GBDT achieved lower MAE than Ridge while avoiding the negative predictions observed for Ridge. The lower error should not be attributed solely to the absence of negative predictions. Within this fitted GBDT model, all observed out-of-fold predictions remained positive, with a minimum of 0.0345 mm. The model also produced a maximum prediction of 0.9790 mm across the 145 audited observations.
   - **High Wear Range ($V_B > 0.40\text{ mm}$, $n = 48$):** Both models suffer from substantial error expansion and underprediction (Ridge MAE = $0.1433\text{ mm}$, Bias = $-0.0495\text{ mm}$; GBDT MAE = $0.1762\text{ mm}$, Bias = $-0.1014\text{ mm}$).
3. **Material Robustness:**  
   On Cast Iron ($n = 97$), both models perform similarly (Ridge MAE = $0.0858\text{ mm}$, GBDT MAE = $0.0839\text{ mm}$). Ridge shows lower prediction error for Stainless Steel J45 in this benchmark, yielding an MAE of $0.1421\text{ mm}$ compared to GBDT's $0.1888\text{ mm}$ ($24.7\%$ lower error).
4. **Condition-Level Consistency:**  
   Across the 8 locked cutting conditions, Ridge achieves lower MAE in 6 conditions, while GBDT achieves lower MAE in 2 conditions (both heavy Cast Iron cuts: C3 and C4).
5. **Governance Confirmation:**  
   The empirical evidence firmly confirms the locked Phase 5.1 architecture: **Ridge Regression remains the Primary Engineering Model** and **GBDT remains the Supporting Nonlinear Model**.

---

## 2. Locked Model Formulations & Benchmarks

The audit evaluates out-of-fold predictions generated from the locked Phase 4 / Phase 5 pipeline on [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv) ($N = 145$ valid runs, 16 tool cases, 8 cutting conditions):

### Model Specifications
- **Primary Engineering Model — Ridge Regression:**
  - Pipeline: `StandardScaler(with_mean=True, with_std=True)` $\rightarrow$ `Ridge(alpha=1.0, random_state=42)`
  - Standardized 8-feature formulation (5 Primary Sensor + 3 Operating Context)
  - Locked Context LOGO: $\text{MAE} = 0.1044\text{ mm}$, $\text{RMSE} = 0.1425\text{ mm}$, $R^2 = 0.6989$
  - Locked Context LOCO: $\text{MAE} = 0.1221\text{ mm}$, $\text{RMSE} = 0.1566\text{ mm}$, $R^2 = 0.6360$
- **Supporting Nonlinear Model — Gradient Tree Boosting (GBDT):**
  - Model: `GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)`
  - Raw 8-feature input (scale-invariant decision trees)
  - Locked Context LOGO: $\text{MAE} = 0.1186\text{ mm}$, $\text{RMSE} = 0.1695\text{ mm}$, $R^2 = 0.5739$
  - Locked Context LOCO: $\text{MAE} = 0.1213\text{ mm}$, $\text{RMSE} = 0.1756\text{ mm}$, $R^2 = 0.5425$

---

## 3. Audit A: Observed vs. Predicted Wear

### 3.1 Ridge Primary Model Behavior
- **Prediction Range:** $[-0.1773, 0.9561]\text{ mm}$ against actual target range $[0.0000, 1.5300]\text{ mm}$.
- **Central Alignment:** Tightly distributed around the 1:1 identity line throughout the $0.15\text{--}0.55\text{ mm}$ range.
- **Negative Extrapolation:** Linear extrapolation produces negative predictions on 8 runs (minimum $-0.1773\text{ mm}$ on fresh tools in heavy cuts with negative context offsets). Overall mean bias is near zero ($-0.0043\text{ mm}$).
- *Artifact:* [Figure 1: Observed vs. Predicted Flank Wear — Ridge](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_ridge_observed_vs_predicted.png)

### 3.2 GBDT Supporting Model Behavior
- **Prediction Range:** $[0.0345, 0.9790]\text{ mm}$.
- **Observed Prediction Range within Fitted Model:** Within this fitted GBDT model, all observed out-of-fold predictions remained positive, with a minimum of 0.0345 mm. The model also produced a maximum prediction of 0.9790 mm across the 145 audited observations.
- **Upper Range Behavior:** The highest predicted value across all 145 runs is $0.9790\text{ mm}$. The fitted GBDT predictions show a lower observed upper range, resulting in underprediction for wear exceeding $0.60\text{ mm}$. Overall mean bias is near zero ($-0.0036\text{ mm}$).
- *Artifact:* [Figure 2: Observed vs. Predicted Flank Wear — GBDT](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_gbdt_observed_vs_predicted.png)

---

## 4. Audit B: Wear-Regime Error Analysis

Observations are grouped into the locked three analytical wear regimes:
1. **Low Wear Range ($V_B < 0.20\text{ mm}$):** $n = 49$ cuts ($33.8\%$), mean wear $0.102\text{ mm}$.
2. **Moderate Wear Range ($0.20 \le V_B \le 0.40\text{ mm}$):** $n = 48$ cuts ($33.1\%$), mean wear $0.295\text{ mm}$.
3. **High Wear Range ($V_B > 0.40\text{ mm}$):** $n = 48$ cuts ($33.1\%$), mean wear $0.626\text{ mm}$.

### Table 1: Wear-Regime Error Comparison (LOGO Context Formulations)

| Wear Regime | Sample Count ($N$) | Target Mean $V_B$ (mm) | Ridge MAE (mm) | Ridge RMSE (mm) | Ridge Bias (mm) | GBDT MAE (mm) | GBDT RMSE (mm) | GBDT Bias (mm) | Superior Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Wear** ($V_B < 0.20\text{ mm}$) | 49 | 0.1020 | 0.0940 | 0.1064 | +0.0060 | **0.0769** | **0.1051** | +0.0430 | **GBDT** |
| **Moderate Wear** ($0.20 \le V_B \le 0.40\text{ mm}$) | 48 | 0.2948 | **0.0761** | **0.0969** | +0.0305 | 0.1035 | 0.1290 | +0.0466 | **Ridge** |
| **High Wear** ($V_B > 0.40\text{ mm}$) | 48 | 0.6260 | **0.1433** | **0.2010** | -0.0495 | 0.1762 | 0.2426 | -0.1014 | **Ridge** |
| **Overall Lifecycle** | **145** | **0.3394** | **0.1044** | **0.1425** | **-0.0043** | **0.1186** | **0.1695** | **-0.0036** | **Ridge** |

*Data serialized at:* [`reports/phase5_2c_wear_regime_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_wear_regime_comparison.csv)  
*Figure:* [Figure 4: Absolute Prediction Error Across the Wear Lifecycle](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_absolute_error_vs_vb.png)

### Key Regime Observations:
1. **Best-Performing Wear Regime (Moderate Wear):** Ridge achieves its lowest error ($\text{MAE} = 0.0761\text{ mm}$, $\text{RMSE} < 0.10\text{ mm}$) in the moderate wear regime ($0.20\text{--}0.40\text{ mm}$), which spans the study's analytical tool-life threshold ($0.30\text{ mm}$). GBDT exhibits higher variance in this regime ($\text{MAE} = 0.1035\text{ mm}$).
2. **Low-Wear Interpretation:** In the low-wear regime, GBDT achieved lower MAE than Ridge while avoiding the negative predictions observed for Ridge. The lower error should not be attributed solely to the absence of negative predictions. Within this fitted GBDT model, all observed out-of-fold predictions remained positive, with a minimum of 0.0345 mm. The model also produced a maximum prediction of 0.9790 mm across the 145 audited observations.
3. **High Wear Error Expansion:** Both models show increasing absolute error at higher observed $V_B$ values, accompanied by negative bias (underprediction). Ridge retains lower underprediction bias ($-0.0495\text{ mm}$ vs. $-0.1014\text{ mm}$) and lower error ($\text{MAE} = 0.1433$ vs. $0.1762\text{ mm}$). This pattern is consistent with the Ridge formulation continuing to produce larger predictions at higher observed wear, while the fitted GBDT predictions show a lower observed upper range.

---

## 5. Audit C: Material-Level Error Behavior

### Table 2: Material Error Comparison

| Material | Tool Count | Run Count ($N$) | Ridge MAE (mm) | Ridge RMSE (mm) | Ridge Bias (mm) | GBDT MAE (mm) | GBDT RMSE (mm) | GBDT Bias (mm) | Error Ratio (GBDT / Ridge) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cast Iron** | 8 | 97 | 0.0858 | 0.1046 | +0.0029 | **0.0839** | 0.1054 | +0.0175 | 0.98 |
| **Stainless Steel J45** | 8 | 48 | **0.1421** | **0.1980** | -0.0187 | 0.1888 | 0.2536 | -0.0462 | 1.33 |

*Data serialized at:* [`reports/phase5_2c_material_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_material_comparison.csv)

### Observations on Material Subgroups:
- **Cast Iron ($n = 97$):** Both models demonstrate comparable, high-fidelity performance ($\text{MAE} \approx 0.084\text{--}0.086\text{ mm}$).
- **Stainless Steel J45 ($n = 48$):** Both models exhibit higher prediction errors on Stainless Steel J45 than on Cast Iron within this benchmark. However, Ridge shows lower prediction error for Stainless Steel J45 in this benchmark: GBDT error expands to $0.1888\text{ mm}$ ($+125\%$ increase over Cast Iron), whereas Ridge error expands to $0.1421\text{ mm}$ ($+65\%$ increase). Ridge shows lower prediction error for Stainless Steel J45 than GBDT in this benchmark.

---

## 6. Audit D: Condition-Level Error Behavior

### Table 3: Condition-Level Error Comparison across 8 Locked Cutting Conditions

| Condition ID | Material | Depth of Cut ($a_p$) | Feed Rate ($f_z$) | Sample Count ($N$) | Ridge MAE (mm) | Ridge Bias (mm) | GBDT MAE (mm) | GBDT Bias (mm) | Better Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Cast Iron | 0.75 mm | 0.25 mm/rev | 34 | **0.0887** | +0.0520 | 0.0914 | +0.0249 | **Ridge** |
| **C2** | Cast Iron | 0.75 mm | 0.50 mm/rev | 24 | **0.0709** | -0.0135 | 0.0917 | +0.0244 | **Ridge** |
| **C3** | Cast Iron | 1.50 mm | 0.25 mm/rev | 17 | 0.0771 | +0.0142 | **0.0493** | +0.0116 | **GBDT** |
| **C4** | Cast Iron | 1.50 mm | 0.50 mm/rev | 22 | 0.1043 | -0.0560 | **0.0905** | +0.0037 | **GBDT** |
| **C5** | Stainless Steel | 0.75 mm | 0.25 mm/rev | 20 | **0.1859** | -0.0430 | 0.2303 | -0.1118 | **Ridge** |
| **C6** | Stainless Steel | 0.75 mm | 0.50 mm/rev | 12 | **0.1187** | -0.0125 | 0.1744 | +0.0152 | **Ridge** |
| **C7** | Stainless Steel | 1.50 mm | 0.25 mm/rev | 7 | **0.0715** | +0.0381 | 0.1627 | +0.0583 | **Ridge** |
| **C8** | Stainless Steel | 1.50 mm | 0.50 mm/rev | 9 | **0.1308** | -0.0284 | 0.1360 | -0.0988 | **Ridge** |

*Data serialized at:* [`reports/phase5_2c_condition_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_condition_comparison.csv)

### Observations on Condition Behavior:
- **Consistently Strong Conditions:** C2 (Cast Iron $0.75\text{ mm}$ / $0.50\text{ mm/rev}$, Ridge MAE = $0.0709\text{ mm}$), C3 (Cast Iron $1.50\text{ mm}$ / $0.25\text{ mm/rev}$, GBDT MAE = $0.0493\text{ mm}$), and C7 (Stainless Steel $1.50\text{ mm}$ / $0.25\text{ mm/rev}$, Ridge MAE = $0.0715\text{ mm}$).
- **Most Difficult Condition:** C5 (Stainless Steel $0.75\text{ mm}$ / $0.25\text{ mm/rev}$, which contains Case 13), where Ridge MAE is $0.1859\text{ mm}$ and GBDT MAE is $0.2303\text{ mm}$.
- **Systematic Condition Comparison:** Ridge outperforms GBDT in **6 out of 8 cutting conditions**. GBDT achieves lower MAE in conditions C3 and C4 (both heavy Cast Iron cuts at $1.50\text{ mm}$ DOC), where tree splits capture localized load-wear relationships.

---

## 7. Audit E: Ridge vs. GBDT Prediction Agreement

### Table 4: Cross-Model Prediction Agreement Metrics ($N = 145$)

| Agreement Metric | Quantitative Value | Unit | Interpretation |
| :--- | :---: | :---: | :--- |
| **Pearson Correlation ($r$)** | **0.8451** | dimensionless | Strong linear concordance between primary and supporting model outputs. |
| **Mean Absolute Difference (MAD)** | **0.0987** | mm | Average absolute divergence between model predictions across all cuts. |
| **RMSE of Prediction Difference** | **0.1222** | mm | Root mean square deviation between model predictions. |
| **Mean Signed Difference ($\text{Ridge} - \text{GBDT}$)** | **-0.0007** | mm | The mean signed difference is close to zero (-0.0007 mm), indicating little average directional offset between the two model predictions across the benchmark. |

*Data serialized at:* [`reports/phase5_2c_model_agreement.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_model_agreement.csv)  
*Figure:* [Figure 3: Cross-Model Prediction Agreement Scatter Plot](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_ridge_vs_gbdt_agreement.png)

### Table 5: Observations with Largest Absolute Disagreement (Top 10)

| Case Audit ID | Run Index | Workpiece Material | DOC (mm) | Feed (mm/rev) | Observed $V_B$ (mm) | Ridge Pred (mm) | GBDT Pred (mm) | Signed Diff (mm) | Absolute Disagreement (mm) | Underlying Disagreement Pattern |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Case 5** | 6 | Stainless Steel J45 | 1.50 | 0.50 | 0.74 | 0.9561 | 0.4790 | +0.4771 | **0.4771** | High wear point; GBDT leaf saturation at $0.48\text{ mm}$ vs. Ridge linear tracking. |
| **Case 15** | 2 | Stainless Steel J45 | 1.50 | 0.25 | 0.15 | 0.2358 | 0.5435 | -0.3077 | **0.3077** | Fresh tool pass; GBDT overpredicts Stainless cut early based on high AE/current. |
| **Case 15** | 4 | Stainless Steel J45 | 1.50 | 0.25 | 0.37 | 0.4105 | 0.6798 | -0.2693 | **0.2693** | Moderate wear pass; GBDT overpredicts wear on Case 15. |
| **Case 7** | 6 | Stainless Steel J45 | 0.75 | 0.25 | 0.34 | 0.4048 | 0.1361 | +0.2687 | **0.2687** | GBDT underpredicts mid-wear cut on Case 7. |
| **Case 14** | 6 | Stainless Steel J45 | 0.75 | 0.50 | 0.35 | 0.4404 | 0.6991 | -0.2587 | **0.2587** | GBDT overpredicts wear on Case 14 cut. |
| **Case 12** | 2 | Cast Iron | 0.75 | 0.50 | 0.05 | -0.1209 | 0.1310 | -0.2520 | **0.2520** | Fresh tool pass; Ridge unconstrained negative extrapolation vs. GBDT positive floor. |
| **Case 9** | 2 | Cast Iron | 1.50 | 0.50 | 0.10 | -0.1773 | 0.0654 | -0.2428 | **0.2428** | Fresh tool pass; Ridge negative extrapolation vs. GBDT positive floor. |
| **Case 9** | 1 | Cast Iron | 1.50 | 0.50 | 0.00 | -0.1727 | 0.0590 | -0.2317 | **0.2317** | Zero wear cut; Ridge negative extrapolation vs. GBDT positive floor. |
| **Case 5** | 2 | Stainless Steel J45 | 1.50 | 0.50 | 0.16 | 0.2564 | 0.4736 | -0.2172 | **0.2172** | Fresh tool pass; GBDT overpredicts Stainless cut. |
| **Case 1** | 1 | Cast Iron | 1.50 | 0.50 | 0.00 | -0.0803 | 0.1259 | -0.2062 | **0.2062** | Zero wear cut; Ridge negative extrapolation vs. GBDT positive floor. |

*Data serialized at:* [`reports/phase5_2c_largest_disagreements.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_largest_disagreements.csv)

### Cross-Model Concordance Synthesis:
> **"The GBDT predictions are broadly aligned with the Ridge predictions (Pearson $r = 0.8451$), indicating that the nonlinear supporting model largely reproduces the main predictive structure captured by the primary linear formulation, while providing a nonlinear cross-check."**

The largest disagreements are concentrated toward the lower and upper parts of the observed wear range, although several moderate-wear observations also show substantial model differences:
1. **Fresh-Tool Wear Range (Low Wear):** Ridge outputs negative values ($-0.08$ to $-0.18\text{ mm}$) due to negative context weights under high feed/DOC, whereas within this fitted GBDT model, all predictions remained positive (above $+0.05\text{ mm}$).
2. **High-Wear Range:** GBDT predictions remain below $0.50\text{ mm}$ on certain high-wear cuts (e.g., Case 5 Run 6, actual $0.74\text{ mm}$), whereas Ridge tracks upward linearly ($0.9561\text{ mm}$).

---

## 8. High-Wear Behavior & Case 13 / VB = 1.53 mm Forensic

### 8.1 Error Escalation in the Upper Wear Tail ($V_B > 0.40\text{ mm}$)
Across all 48 observations in the high-wear range, prediction error increases systematically:
- Ridge MAE increases from $0.0761\text{ mm}$ (moderate wear) to **$0.1433\text{ mm}$** (high wear).
- GBDT MAE increases from $0.1035\text{ mm}$ (moderate wear) to **$0.1762\text{ mm}$** (high wear).
- Both models systematically underpredict in this tail (Ridge bias = $-0.0495\text{ mm}$, GBDT bias = $-0.1014\text{ mm}$).

> *"Prediction error increases toward the upper observed wear range, with limited evidence for reliable extrapolation beyond the central training range."*

### 8.2 Extreme Observation ($V_B = 1.53\text{ mm}$, Case 13 Run 15)
The single highest wear point in the dataset ($V_B = 1.53\text{ mm}$) illustrates the severe extrapolation limit:
- **Observed Wear:** $1.5300\text{ mm}$
- **Ridge Prediction:** $0.7902\text{ mm}$ (Error = $-0.7398\text{ mm}$)
- **GBDT Prediction:** $0.8523\text{ mm}$ (Error = $-0.6777\text{ mm}$)

> *"The highest observed wear point (`VB = 1.53 mm`) contributes disproportionately to high-wear prediction error."*

In Case 13, model errors remained below $0.03\text{ mm}$ while flank wear remained under $0.32\text{ mm}$ (Runs 3–6). Once flank wear escalated past $0.40\text{ mm}$, both models progressively lagged behind actual wear, confirming that neither formulation can reliably extrapolate to severe, accelerated wear states.

---

## 9. Final Synthesis: Addressing the Five Core Review Questions

### Q1. Where does Ridge perform best?
**Ridge performs best in the Moderate Wear regime ($0.20 \le V_B \le 0.40\text{ mm}$)**, achieving an MAE of **$0.0761\text{ mm}$** ($\text{RMSE} = 0.0969\text{ mm}$). This critical operating window encompasses the study's analytical tool-life threshold ($V_B = 0.30\text{ mm}$). Ridge shows lower prediction error for Stainless Steel J45 in this benchmark ($\text{MAE} = 0.1421\text{ mm}$) and achieves lower error in 6 out of 8 cutting conditions.

### Q2. Where does GBDT perform best?
**GBDT performs best in the Low Wear regime ($V_B < 0.20\text{ mm}$)**. In the low-wear regime, GBDT achieved lower MAE than Ridge while avoiding the negative predictions observed for Ridge. The lower error should not be attributed solely to the absence of negative predictions. Within this fitted GBDT model, all observed out-of-fold predictions remained positive, with a minimum of 0.0345 mm. The model also produced a maximum prediction of 0.9790 mm across the 145 audited observations. GBDT achieved an MAE of **$0.0769\text{ mm}$** compared to Ridge's $0.0940\text{ mm}$ (which unconstrainedly predicts into negative territory). GBDT also achieves lower prediction error in heavy Cast Iron cutting conditions (C3: MAE $0.0493\text{ mm}$, C4: MAE $0.0905\text{ mm}$).

### Q3. Do both models share the same major weaknesses?
**Yes.** Both models share two identical structural limitations:
1. **High-Wear Underprediction:** Both models fail to extrapolate to severe wear states ($V_B > 0.40\text{ mm}$), exhibiting substantial negative bias and error expansion, culminating in severe underprediction at the $V_B = 1.53\text{ mm}$ extreme point.
2. **Material Discrepancy:** Both models exhibit higher prediction errors on Stainless Steel J45 than on Cast Iron within this benchmark.

### Q4. Are Ridge and GBDT predictions broadly aligned?
**Yes.** Ridge and GBDT predictions are strongly concordant ($\text{Pearson } r = 0.8451$, $\text{MAD} = 0.0987\text{ mm}$, Net Mean Signed Difference = $-0.0007\text{ mm}$). The mean signed difference is close to zero (-0.0007 mm), indicating little average directional offset between the two model predictions across the benchmark. The supporting GBDT model confirms the primary linear predictive structure rather than contradicting it. The largest disagreements are concentrated toward the lower and upper parts of the observed wear range, although several moderate-wear observations also show substantial model differences.

### Q5. Does the evidence justify keeping Ridge = Primary Engineering Model and GBDT = Supporting Nonlinear Model?
**Yes. The audit supports retaining Ridge as the Primary Engineering Model and GBDT as the Supporting Nonlinear Model within this benchmark.** The audit provides decisive empirical support for the locked dual-model architecture:
- **Ridge as Primary Engineering Model:** Maintains higher overall accuracy (LOGO MAE $0.1044$ vs. $0.1186\text{ mm}$, $R^2 = 0.6989$ vs. $0.5739$), superior performance in the study threshold window ($0.20\text{--}0.40\text{ mm}$, MAE $0.0761$ vs. $0.1035\text{ mm}$), greater robustness across cutting conditions (better in 6/8 conditions), lower prediction error on difficult materials (Stainless Steel MAE $0.1421$ vs. $0.1888\text{ mm}$), and direct coefficient-based interpretability of the fitted linear formulation.
- **GBDT as Supporting Nonlinear Model:** GBDT serves as a nonlinear model-form cross-check of the primary Ridge formulation, preventing negative fresh-tool predictions, providing a cross-model interpretation check of the relative predictive importance of the sensor features, and capturing non-linear vibration peakedness sensitivity.

---

## 10. Methodological Limitations

1. **Benchmark Scope:**  
   The final model-behavior audit is based on the locked validation framework and the observed NASA Milling benchmark. It does not establish performance on unseen machines, unseen materials, unseen sensor configurations, or operating conditions outside the tested design.
2. **Non-Nested Feature Screening Boundary:**  
   The feature screening step was performed once using all labeled observations; therefore, the overall validation is not a fully nested model-development estimate.
3. **Absence of Significance Testing:**  
   In accordance with the project governance, no formal hypothesis testing, t-tests, or p-values were computed. All comparative evaluations are strictly descriptive and diagnostic.
4. **No Replacement Reliability Claim:**  
   While Ridge achieves low error in the moderate wear window surrounding $0.30\text{ mm}$, tool replacement decision reliability under industrial production tolerances was not directly tested.

---

## 11. Deliverables & Data Manifest

- 📊 **Wear-Regime Comparison CSV:** [`reports/phase5_2c_wear_regime_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_wear_regime_comparison.csv)
- 📊 **Material Comparison CSV:** [`reports/phase5_2c_material_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_material_comparison.csv)
- 📊 **Condition Comparison CSV:** [`reports/phase5_2c_condition_comparison.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_condition_comparison.csv)
- 📊 **Cross-Model Agreement CSV:** [`reports/phase5_2c_model_agreement.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_model_agreement.csv)
- 📊 **Top 10 Largest Disagreements CSV:** [`reports/phase5_2c_largest_disagreements.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_2c_largest_disagreements.csv)
- 🖼️ **Figure 1 (Ridge Observed vs. Predicted):** [Figure 1: Observed vs. Predicted Flank Wear — Ridge](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_ridge_observed_vs_predicted.png)
- 🖼️ **Figure 2 (GBDT Observed vs. Predicted):** [Figure 2: Observed vs. Predicted Flank Wear — GBDT](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_gbdt_observed_vs_predicted.png)
- 🖼️ **Figure 3 (Ridge vs. GBDT Agreement):** [Figure 3: Cross-Model Prediction Agreement](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_ridge_vs_gbdt_agreement.png)
- 🖼️ **Figure 4 (Absolute Error across Wear Lifecycle):** [Figure 4: Absolute Prediction Error Across the Wear Lifecycle](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2c_absolute_error_vs_vb.png)
- 🐍 **Execution & Audit Script:** [`scripts/phase5_2c_final_model_behavior_audit.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase5_2c_final_model_behavior_audit.py)

---

## 12. Phase 5.2C Governance Statement

> **"This phase performed a descriptive robustness and model-behavior audit of the locked Ridge Primary Model and GBDT Supporting Model. No model, feature, hyperparameter, target definition, or validation strategy was changed. The analysis identifies where prediction errors are concentrated and quantifies agreement between the two locked formulations. Ridge remains the Primary Engineering Model and GBDT remains the Supporting Nonlinear Model."**

---

*Report certified by AntiGravity (Implementation Agent) for review by Strategic Lead ChatGPT and Domain Lead Bright.*
