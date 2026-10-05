# Phase 5.2A: Final Ridge Formulation & Coefficient Audit
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2) — CNC Machining & Tool Wear Analytics
- **Document Type:** Final Primary Model Verification & Coefficient Audit
- **Author:** AntiGravity (Implementation Agent)
- **Reviewers:** Bright (Domain Lead) & ChatGPT (Strategic Lead)
- **Status:** Phase 5.2A Complete (Wording Revisions Locked)

---

## 1. Executive Summary

Phase 5.2A executes the implementation audit and coefficient extraction for the primary engineering model selected in Phase 5.1:
> **Ridge Regression + Context Fusion (`alpha = 1.0`, Target = `VB_mm`)**

### Core Findings:
1. **Implementation Verification:**  
   The pipeline implementation strictly matches the locked specification: 5 Primary Sensor Features + 3 Operating Context Features, preprocessed via `StandardScaler()` strictly fitted on training splits, with zero data leakage.
2. **Benchmark Reproduction:**  
   The locked Phase 4 benchmark metrics are reproduced identically to 4 decimal places:
   - **Context LOGO (16 folds):** $\text{MAE} = 0.1044\text{ mm}$, $\text{RMSE} = 0.1425\text{ mm}$, $R^2 = 0.6989$.
   - **Context LOCO (8 folds):** $\text{MAE} = 0.1221\text{ mm}$, $\text{RMSE} = 0.1566\text{ mm}$, $R^2 = 0.6360$.
3. **Fitted Model Weighting:**  
   In standardized feature space, `smcAC_rms` has the largest standardized coefficient ($\beta = +0.2940$) in the final Ridge formulation, indicating that the model places the greatest linear weight on this sensor feature. Within the fitted multivariable formulation, `DOC_mm` ($\beta = -0.1547$) and `feed_mm_rev` ($\beta = -0.0944$) receive negative coefficients while `smcAC_rms` receives a larger positive coefficient. This indicates that the model uses operating-condition information jointly with spindle current when estimating $V_B$; the coefficients should not be interpreted as a physical decomposition of cutting load and wear-related current.
4. **Methodological Integrity:**  
   No non-negative clipping was applied during evaluation. No hyperparameter tuning or feature alterations were performed.

---

## 2. Implementation Audit

### 2.1 Relevant Files
- **Dataset:** [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv) (145 valid cuts, 16 tool cases, 8 conditions).
- **Execution Script:** [`scripts/phase5_2a_ridge_audit.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase5_2a_ridge_audit.py).
- **Prior Reference Implementations:**  
  - [`scripts/phase4_1_3_ml_benchmark.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase4_1_3_ml_benchmark.py) (Sensor-only LOGO).
  - [`scripts/phase4_5_context_fusion.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase4_5_context_fusion.py) (Context-fused LOGO).
  - [`scripts/phase4_5B_LOCO.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase4_5B_LOCO.py) (Context-fused LOCO).

### 2.2 Feature Set Audit
The 8 features input to the final model are:
- **Primary Sensor Features (5):**
  1. `smcAC_rms`
  2. `vib_spindle_kurtosis`
  3. `vib_spindle_p2p`
  4. `AE_table_rms`
  5. `AE_spindle_p2p`
- **Operating Context Features (3):**
  6. `material_code`
  7. `DOC_mm`
  8. `feed_mm_rev`
- **Target:** `VB_mm` (Continuous flank wear land width).

### 2.3 Preprocessing & Standardization Audit
- **Pipeline Structure:** The model is formulated as an encapsulated scikit-learn Pipeline:
  ```python
  Pipeline([
      ('scaler', StandardScaler(with_mean=True, with_std=True)),
      ('model', Ridge(alpha=1.0, random_state=42))
  ])
  ```
- **Standardization:** All 8 numerical and encoded features are centered to mean 0 and scaled to unit variance ($Z = (X - \mu)/\sigma$).
- **Leakage Audit:** In cross-validation (both LOGO and LOCO), `StandardScaler` is fitted strictly on the training fold (`X_train`) and transforms the held-out test fold (`X_test`). Zero out-of-fold information leaks into preprocessing.
- **Material Encoding:**  
  `material_code` in `feature_dataset_v32.csv` is an integer mapping:
  - `material_code = 1` $\rightarrow$ **Cast Iron** ($n = 97$ cuts, $66.9\%$).
  - `material_code = 2` $\rightarrow$ **Stainless Steel J45** ($n = 48$ cuts, $33.1\%$).  
  When passed into `StandardScaler()`, Cast Iron maps to a negative value ($Z \approx -0.70$) and Stainless Steel J45 maps to a positive value ($Z \approx +1.42$).

---

## 3. Benchmark Reproduction

Using the verified scikit-learn pipeline, the locked Phase 4 benchmark metrics were reproduced without discrepancy:

### Table 1: Benchmark Reproduction Audit

| Benchmark Evaluation | Metric | Reproduced Value | Locked Phase 4 Reference | Discrepancy | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Context LOGO** | **MAE (mm)** | **0.1044** | 0.1044 | 0.0000 | **Verified Exact** |
| (16 Tool Folds) | **RMSE (mm)** | **0.1425** | 0.1425 | 0.0000 | **Verified Exact** |
| | **$R^2$** | **0.6989** | 0.6989 | 0.0000 | **Verified Exact** |
| **Context LOCO** | **MAE (mm)** | **0.1221** | 0.1221 | 0.0000 | **Verified Exact** |
| (8 Condition Folds) | **RMSE (mm)** | **0.1566** | 0.1566 | 0.0000 | **Verified Exact** |
| | **$R^2$** | **0.6360** | 0.6360 | 0.0000 | **Verified Exact** |

*Takeaway:* Zero divergence from locked Phase 4 values.

---

## 4. Final Ridge Coefficients (Standardized Feature Space)

When fitted on the full valid dataset ($N = 145$ cuts, 16 tool cases) with standardized features and $\alpha = 1.0$:
- **Model Intercept ($\beta_0$):** $+0.3394\text{ mm}$ (identically equal to the unweighted mean of $V_B$).

### Table 2: Complete 8-Feature Standardized Coefficient Table

| Rank | Feature | Type | Standardized Coefficient ($\beta$) | Sign | Absolute Magnitude ($|\beta|$) | LOGO Fold Mean $\pm$ Std | LOCO Fold Mean $\pm$ Std |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `smcAC_rms` | Sensor | **+0.2940** | $+$ | 0.2940 | $+0.2915 \pm 0.0246$ | $+0.2881 \pm 0.0378$ |
| **2** | `DOC_mm` | Context | **-0.1547** | $-$ | 0.1547 | $-0.1534 \pm 0.0114$ | $-0.1532 \pm 0.0189$ |
| **3** | `feed_mm_rev` | Context | **-0.0944** | $-$ | 0.0944 | $-0.0935 \pm 0.0086$ | $-0.0933 \pm 0.0149$ |
| **4** | `AE_spindle_p2p` | Sensor | **+0.0302** | $+$ | 0.0302 | $+0.0303 \pm 0.0050$ | $+0.0297 \pm 0.0075$ |
| **5** | `vib_spindle_kurtosis` | Sensor | **+0.0205** | $+$ | 0.0205 | $+0.0203 \pm 0.0037$ | $+0.0209 \pm 0.0061$ |
| **6** | `AE_table_rms` | Sensor | **-0.0199** | $-$ | 0.0199 | $-0.0188 \pm 0.0102$ | $-0.0158 \pm 0.0153$ |
| **7** | `vib_spindle_p2p` | Sensor | **-0.0156** | $-$ | 0.0156 | $-0.0154 \pm 0.0049$ | $-0.0167 \pm 0.0078$ |
| **8** | `material_code` | Context | **+0.0056** | $+$ | 0.0056 | $+0.0056 \pm 0.0046$ | $+0.0048 \pm 0.0115$ |

*Data serialized at:* [`reports/phase5_final_ridge_coefficients.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_final_ridge_coefficients.csv)  
*Figure:* [Figure: Final Context-Fused Ridge Regression Coefficients](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2a_ridge_coefficients.png)

---

## 5. Sensor-Only vs. Context-Fused Coefficient Audit

To document how model weighting changes when process variables are introduced, Table 3 compares the Sensor-Only Ridge model (5 features) against the Final Context-Fused Ridge model (8 features):

### Table 3: Shift in Standardized Coefficients Upon Adding Operating Context

| Sensor Feature | Sensor-Only Coefficient ($\beta_{\text{sensor}}$) | Context-Fused Coefficient ($\beta_{\text{context}}$) | Shift ($\Delta \beta$) | Observation on Fitted Model Weighting |
| :--- | :---: | :---: | :---: | :--- |
| `smcAC_rms` | $+0.1814$ | **$+0.2940$** | **$+0.1127$** | Increased weight after accounting for process variables |
| `AE_table_rms` | $-0.0806$ | **$-0.0199$** | $+0.0607$ | Attenuated weight in presence of context |
| `vib_spindle_kurtosis` | $+0.0746$ | **$+0.0205$** | $-0.0541$ | Positive peakedness indicator |
| `AE_spindle_p2p` | $+0.0701$ | **$+0.0302$** | $-0.0399$ | High-frequency contact pulse amplitude |
| `vib_spindle_p2p` | $+0.0065$ | **$-0.0156$** | $-0.0220$ | Small negative weight near zero |

### Explanation of Coefficient Shift:
- With operating context included in the model, the fitted Ridge formulation assigns a larger positive coefficient to `smcAC_rms` ($+0.1814 \rightarrow +0.2940$), indicating that spindle-current variation carries stronger predictive weight for $V_B$ after accounting for the included process variables.
- Within the fitted multivariable formulation, `DOC_mm` and `feed_mm_rev` receive negative coefficients while `smcAC_rms` receives a larger positive coefficient. This reflects a shift in fitted model weighting under joint estimation, and should not be interpreted as a physical decomposition of cutting load and wear-related current.

---

## 6. Interpretation & Conservative Synthesis

### 6.1 Sensor Features
- **`smcAC_rms` ($\beta = +0.2940$, Rank 1):**  
  `smcAC_rms` has the largest standardized coefficient in the final Ridge formulation, indicating that the model places the greatest linear weight on this sensor feature after accounting for the included process variables.
- **`AE_spindle_p2p` ($\beta = +0.0302$, Rank 4) and `vib_spindle_kurtosis` ($\beta = +0.0205$, Rank 5):**  
  Both carry positive weights, indicating that higher acoustic emission transient amplitudes and higher spindle vibration signal peakedness contribute positively to estimated flank wear in the fitted linear model.
- **`AE_table_rms` ($\beta = -0.0199$, Rank 6) and `vib_spindle_p2p` ($\beta = -0.0156$, Rank 7):**  
  Both exhibit small negative weights of low magnitude ($|\beta| < 0.02$), functioning as secondary adjustments within the regularized multi-collinear sensor space.

### 6.2 Operating Context Features
- **`DOC_mm` ($\beta = -0.1547$, Rank 2) and `feed_mm_rev` ($\beta = -0.0944$, Rank 3):**  
  Within the fitted multivariable formulation, DOC and feed receive negative coefficients while `smcAC_rms` receives a larger positive coefficient. This indicates that the model uses operating-condition information jointly with spindle current when estimating $V_B$; the coefficients should not be interpreted as a physical decomposition of cutting load and wear-related current, nor do they represent a claim that heavier cuts physically reduce wear.
- **`material_code` ($\beta = +0.0056$, Rank 8):**  
  `material_code` carries a small positive weight ($+0.0056$). Under the encoded mapping ($1 = \text{Cast Iron}$, $2 = \text{Stainless Steel J45}$), Stainless Steel cuts receive a slight positive baseline adjustment relative to Cast Iron cuts in standardized space, although its magnitude is the smallest of all 8 features.

### 6.3 What Can and Cannot Be Inferred
- **Can Be Inferred:**  
  In the fitted linear formulation, `smcAC_rms` has the largest standardized coefficient, and operating conditions (DOC and feed) enter jointly with negative coefficients. The coefficient values are highly stable across 16 LOGO folds (`smcAC_rms`: $+0.2915 \pm 0.0246$) and 8 LOCO folds ($+0.2881 \pm 0.0378$).
- **Cannot Be Inferred:**  
  The negative coefficients for DOC and feed rate must not be interpreted as physical causes of reduced tool wear. They reflect parameter weights within a joint multi-variable regression model where features are mutually correlated. Furthermore, standardized coefficient magnitude should not be quoted as an additive percentage contribution of wear.

---

## 7. Deliverables & Data Manifest

- 📊 **Coefficient Table CSV:** [`reports/phase5_final_ridge_coefficients.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/phase5_final_ridge_coefficients.csv)
- 🖼️ **Coefficient Plot Figure:** [Figure: Final Context-Fused Ridge Regression Coefficients](file:///D:/Project/Manufacturing%20Analytics/reports/figures/fig_phase5_2a_ridge_coefficients.png)
- 🐍 **Verification Script:** [`scripts/phase5_2a_ridge_audit.py`](file:///D:/Project/Manufacturing%20Analytics/scripts/phase5_2a_ridge_audit.py)

---

## 8. Governance & Methodological Integrity Statement

> **"No methodological discrepancy identified. The implementation strictly matches the locked Phase 4 specifications. Both Context LOGO and Context LOCO benchmark results were reproduced with zero error. No non-negative clipping was applied, no SHAP calculations were performed, and no Phase 4 metrics were modified."**

---

*Report certified by AntiGravity (Implementation Agent) for review by Strategic Lead ChatGPT.*
