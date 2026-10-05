# V2 Dataset Deep Audit: NASA Milling Tool Wear Dataset

**Project:** CNC Predictive Maintenance & Tool Degradation Analytics (V2)  
**Investigator:** AntiGravity (Implementation Agent)  
**Audience / Strategic Lead:** ChatGPT (Strategic Lead) & Bright (Project Owner)  
**Date:** September 2026  
**Dataset Source:** NASA Prognostics Center of Excellence (PCoE) / UC Berkeley BEST Lab (Goebel & Agogino, 1999)  
**Source URL:** `https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip`  
**Raw Files Inspected:** `mill.mat` (15.1 MB), `Readme.pdf` (10 pages)

---

## 1. Dataset Overview & Physical Setup

The **NASA Milling Wear Dataset** captures experimental face milling operations performed on a **Matsuura MC-510V 3-axis CNC Vertical Machining Center**. The experiment investigated gradual cutting tool wear under systematic industrial cutting conditions.

* **Cutting Tool:** KC710 coated carbide inserts mounted on a face milling cutter.
* **Workpiece Dimensions:** 483 mm × 178 mm × 51 mm.
* **Workpiece Materials:**
  * Material 1: **Cast Iron** (8 cases)
  * Material 2: **Stainless Steel J45** (8 cases)
* **Cutting Speed:** Kept constant at **200 m/min** ($\approx$ 826 RPM spindle speed).
* **Operating Matrix:** 2 Depths of Cut (0.75 mm, 1.50 mm) × 2 Feeds (0.25 mm/rev, 0.50 mm/rev) × 2 Materials = **8 condition combinations**.
* **Replication:** Each of the 8 settings was repeated twice using fresh sets of inserts, yielding **16 experimental Cases (Tool Inserts)**.

---

## 2. Raw Data Structure

The dataset is distributed as a single MATLAB file `mill.mat` containing a structured array `mill` of dimension **1 × 167** (167 individual milling passes/runs).

```text
mill (1 x 167 struct array)
├── case         : Experimental case / Tool ID (1 to 16)
├── run          : Pass counter within the case (1 to N)
├── VB           : Flank wear measurement (mm) [periodic ground truth]
├── time         : Cumulative machining duration (minutes)
├── DOC          : Depth of cut (mm)
├── feed         : Feed rate (mm/rev)
├── material     : Workpiece material code (1=Cast Iron, 2=Steel)
├── smcAC        : AC spindle motor current (waveform)
├── smcDC        : DC spindle motor current (waveform)
├── vib_table    : Table vibration (waveform)
├── vib_spindle  : Spindle vibration (waveform)
├── AE_table     : Acoustic emission at table (waveform)
└── AE_spindle   : Acoustic emission at spindle (waveform)
```

### Signal Lengths and Sampling
* **Sampling Frequency ($f_s$):** **250 Hz** (analog pre-processing with RMS converter $\Delta T = 8.0\text{ ms}$).
* **Signal Dimension per Run:**
  * **166 of 167 runs** have exactly **9,000 samples** ($9,000 / 250\text{ Hz} = 36.0\text{ seconds}$).
  * **1 single run** (Case 12, Run 1) has **15,360 samples** ($15,360 / 250\text{ Hz} = 61.44\text{ seconds}$).
  * *Cause:* DAQ was left recording for an extended duration before/after the cut during that specific pass.
* **Sensors per Run:** 6 continuous channels recorded synchronously.

---

## 3. Variables & Attributes

| Variable | Physical Meaning | Unit | Type | Range | Time-Varying? | Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `case` | Tool Insert / Experiment ID | - | Integer | 1 – 16 | Static per tool | Identifier |
| `run` | Cutting pass number | - | Integer | 1 – 23 | Discrete increment | Counter |
| `time` | Cumulative machining time | min | Continuous | 1.0 – 89.0 | Cumulative monotonic | Temporal Tracker |
| `DOC` | Depth of cut | mm | Categorical | 0.75 or 1.50 | Static per case | Operating Parameter |
| `feed` | Feed rate per revolution | mm/rev | Categorical | 0.25 or 0.50 | Static per case | Operating Parameter |
| `material` | Workpiece material | - | Binary | 1 (Cast Iron), 2 (Steel) | Static per case | Workpiece Property |
| `VB` | Flank wear width | mm | Continuous | 0.00 – 1.53 | Monotonic increasing | **Target Variable** |
| `smcAC` | AC Spindle Motor Current | A | Time-Series | -4.20 to +3.82 | Continuous (250 Hz) | Physical Sensor |
| `smcDC` | DC Spindle Motor Current | V | Time-Series | +0.62 to +7.82 | Continuous (250 Hz) | Physical Sensor |
| `vib_table` | Vibration on Table | g | Time-Series | +0.06 to +2.81 | Continuous (250 Hz) | Physical Sensor |
| `vib_spindle` | Vibration on Spindle | g | Time-Series | +0.28 to +0.86 | Continuous (250 Hz) | Physical Sensor |
| `AE_table` | Acoustic Emission on Table | V | Time-Series | +0.07 to +0.47 | Continuous (250 Hz) | Physical Sensor |
| `AE_spindle` | Acoustic Emission on Spindle | V | Time-Series | +0.08 to +0.43 | Continuous (250 Hz) | Physical Sensor |

---

## 4. Sensor Information & Physical Degradation Mechanism

1. **Acoustic Emission (`AE_table`, `AE_spindle`):**
   * *Physical Source:* High-frequency stress wave emission ($50\text{ kHz} - 2\text{ MHz}$) generated during plastic deformation in the primary shear zone, sliding friction along the tool-chip interface, and flank-workpiece friction.
   * *Pre-processing:* Amplified, filtered ($1\text{ kHz} - 8\text{ kHz}$ bandpass), passed through analog RMS detector ($\Delta T = 8\text{ ms}$).
   * *Degradation Relevance:* As flank wear lands expand ($V_B > 0.3\text{ mm}$), contact friction escalates dramatically, increasing RMS AE energy.
2. **Vibration Sensors (`vib_table`, `vib_spindle`):**
   * *Physical Source:* Accelerometers measuring dynamic cutting force oscillations ($0 - 40\text{ kHz}$).
   * *Pre-processing:* Low-pass filtered at 400 Hz (rejecting 180 Hz 3rd power harmonic), high-pass at 1 kHz, RMS converted.
   * *Degradation Relevance:* Duller cutting edges alter tool engagement dynamics, leading to chatter, periodic surface impacts, and increased table/spindle acceleration variance.
3. **Motor Currents (`smcAC`, `smcDC`):**
   * *Physical Source:* Current transducer on spindle motor power phase. Direct indirect proxy for spindle cutting torque ($P = \tau \cdot \omega$).
   * *Degradation Relevance:* As tool edge radius rounds and flank wear rubs against the workpiece, required tangential cutting force and spindle load increase monotonically.

---

## 5. Tool Wear ($V_B$) Measurement & Ground Truth Audit

### How $V_B$ Was Measured
* Ground truth tool wear was **NOT measured in-situ or continuously**.
* Between cutting passes, the machining process was halted, the insert was physically removed from the cutter body, and flank wear width ($V_B$) was inspected under an optical microscope in millimeters.
* Measurements were taken at **irregular intervals**.

### Completeness & Missing Values
* **Total Runs:** 167
* **Runs with measured $V_B$:** **146 runs** (87.4%)
* **Runs with missing $V_B$ (`NaN`):** **21 runs** (12.6%)
  * *Pattern of Missingness:* Missing measurements occur in initial passes (e.g. Case 13, 14, 15, 16 Run 1 where wear was assumed 0.0 but not recorded) and occasional intermediate passes where the technician did not pull the insert.
* **$V_B$ Statistics (across 146 labeled runs):**
  * Min: $0.00\text{ mm}$ (Fresh insert)
  * 25%: $0.15\text{ mm}$
  * Median (50%): $0.285\text{ mm}$
  * 75%: $0.468\text{ mm}$
  * Max: $1.530\text{ mm}$ (Severe wear far beyond the analytical threshold)
  * Analytical tool-life threshold: $V_B = 0.30\text{ mm}$, motivated by the uniform flank-wear criterion in ISO 8688-2 for milling.

### Tool Life Across the 16 Cases

| Case | Total Passes | Labeled Passes | $V_B$ Range (mm) | DOC (mm) | Feed (mm/rev) | Material |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 17 | 13 | 0.00 – 0.50 | 1.50 | 0.50 | Cast Iron |
| **2** | 14 | 13 | 0.08 – 0.55 | 0.75 | 0.50 | Cast Iron |
| **3** | 14 | 14 | 0.00 – 0.55 | 0.75 | 0.25 | Cast Iron |
| **4** | 7 | 7 | 0.08 – 0.49 | 1.50 | 0.25 | Cast Iron |
| **5** | 6 | 6 | 0.00 – 0.74 | 1.50 | 0.50 | Steel J45 |
| **6** | 1 | 1 | 0.00 – 0.00 | 1.50 | 0.25 | Steel J45 (Aborted after 1 cut) |
| **7** | 8 | 7 | 0.00 – 0.46 | 0.75 | 0.25 | Steel J45 |
| **8** | 6 | 5 | 0.00 – 0.62 | 0.75 | 0.50 | Steel J45 |
| **9** | 9 | 9 | 0.00 – 0.81 | 1.50 | 0.50 | Cast Iron |
| **10** | 10 | 10 | 0.00 – 0.70 | 1.50 | 0.25 | Cast Iron |
| **11** | 23 | 20 | 0.00 – 0.76 | 0.75 | 0.25 | Cast Iron |
| **12** | 15 | 12 | 0.05 – 0.65 | 0.75 | 0.50 | Cast Iron |
| **13** | 15 | 13 | 0.10 – 1.53 | 0.75 | 0.25 | Steel J45 |
| **14** | 9 | 7 | 0.09 – 1.14 | 0.75 | 0.50 | Steel J45 |
| **15** | 7 | 6 | 0.15 – 0.70 | 1.50 | 0.25 | Steel J45 |
| **16** | 6 | 3 | 0.24 – 0.62 | 1.50 | 0.50 | Steel J45 |

---

## 6. Generated Visualizations

Four audit figures have been generated and saved to `V2_NASA_Milling/reports/figures/`:

1. **`fig1_tool_wear_progression.png`**: Plots the wear curve ($V_B$ vs Run #) across all 16 tool inserts. Confirms non-linear progressive wear (break-in, steady-state, severe tertiary wear) and shows high wear acceleration in Steel J45 compared to Cast Iron.
2. **`fig2_sensor_signals_fresh_vs_worn.png`**: Direct visual comparison of all 6 raw signals (Case 1 Run 1 vs Run 17). Demonstrates clearly visible amplitude jumps in DC motor current, table vibration RMS, and acoustic emission as tool wears from 0.00 mm to 0.50 mm.
3. **`fig3_tool_life_by_condition.png`**: Bar chart showing tool life (number of cutting passes) grouped by operating conditions and material. Confirms cutting Stainless Steel exhausts tool life in 6–9 passes, whereas Cast Iron sustains up to 23 passes.
4. **`fig4_sensor_features_vs_wear.png`**: Multi-scatter of extracted RMS / Mean sensor features against $V_B$, showing that sensor response curves differ fundamentally between Cast Iron and Stainless Steel.

---

## 7. Data Leakage Audit

A strict categorization of all variables was conducted to avoid the methodological pitfalls identified in V1:

| Variable | Leakage Classification | Rationale / Rule |
| :--- | :--- | :--- |
| `run` | **DEFINITE LEAKAGE** | Monotonically increments per tool. If included in ML models, tree models simply learn "wear $\approx f(\text{run})$", bypassing sensor physics entirely. Must be **excluded** from feature set. |
| `time` | **DEFINITE LEAKAGE** | Cumulative machining minutes. Strongly collinear with cumulative wear. In production, elapsed time without sensor feedback fails if cutting hardness fluctuates. Must be **excluded** from feature set. |
| `case` | **DEFINITE LEAKAGE** | Categorical insert identifier. Must only be used as a **grouping key for cross-validation**, never as an input feature. |
| `smcAC`, `smcDC` | **SAFE** | Physical sensor streaming during the current cut. Completely available at inference time. |
| `vib_table`, `vib_spindle` | **SAFE** | Real-time vibration accelerations during the cut. |
| `AE_table`, `AE_spindle` | **SAFE** | Real-time acoustic emissions during the cut. |
| `DOC`, `feed`, `material` | **SAFE** | Known operational parameters pre-set on the CNC controller prior to cutting. Safe to use as operating context. |

---

## 8. Time / Run Structure & Validation Strategy

* **Observation Unit:** Repeated cutting passes on 16 distinct tool inserts.
* **Why Random 80/20 Train-Test Split is FLAWED:**
  * If Pass 3 and Pass 4 of Case 1 are randomly assigned to Train and Test sets, the test set has near-identical tool micro-geometry and cutting conditions. The model would evaluate data memorization, not generalization.
* **Technically Defensible Validation Strategies:**
  1. **Leave-One-Tool-Out (LOTO) / GroupKFold by `case`:**
     * Group by `case` (16 groups). Train on 12–14 tool inserts, evaluate on unseen tool inserts.
     * This directly answers: *"Can the model estimate wear on a new, unseen cutting tool?"*
  2. **Leave-Condition-Out (Cross-Condition Generalization):**
     * Train on Cast Iron (Material 1), test on Steel (Material 2), or train on DOC=0.75 mm and test on DOC=1.50 mm.
     * Tests model domain adaptability under parameter shifts.

---

## 9. Evaluation of Candidate ML Problems

### Candidate A: Continuous Tool-Wear Regression ($V_B$ in mm)
* **Target:** Continuous $V_B$ (146 labeled passes).
* **Feasibility:** **HIGH**.
* **Scientific Merit:** Direct physical prediction of flank wear width using feature extraction from sensor waveforms (RMS, Peak, Crest Factor, Spectral Centroid, FFT energy).
* **Limitations:** 146 observations is a small sample size for tabular ML. Requires careful feature engineering and strict regularized models (Ridge, Random Forest, XGBoost with small depth). Deep learning (LSTM/CNN) would heavily overfit.
* **Portfolio Value:** **VERY HIGH**. Shows real signal processing, physics understanding, and disciplined evaluation without synthetic inflation.

### Candidate B: Wear-Stage Classification (3-Class or Binary)
* **Target:**
  * 3-Class: Sharp ($V_B < 0.2\text{ mm}$), Normal ($0.2 \le V_B < 0.4\text{ mm}$), Severe ($V_B \ge 0.4\text{ mm}$).
  * Binary: Normal ($V_B < 0.3\text{ mm}$) vs Replaced/Worn ($V_B \ge 0.3\text{ mm}$) per ISO 8688-2.
* **Feasibility:** **VERY HIGH**.
* **Scientific Merit:** Aligns directly with industrial shop-floor decision making (Stop machine / Change insert vs Proceed).
* **Limitations:** Threshold selection must be justified using machining standards (e.g. ISO 8688-2), not arbitrary quantiles.
* **Portfolio Value:** **HIGH**. Extremely intuitive for manufacturing engineers and recruiters.

### Candidate C: Unsupervised / Semi-Supervised Degradation Detection
* **Target:** Anomaly score or degradation distance (e.g. Mahalanobis distance, Isolation Forest, Reconstruction error from fresh tool baseline).
* **Feasibility:** **MEDIUM**.
* **Scientific Merit:** Uses all 167 runs (including 21 unlabeled runs). Evaluates distance from initial Pass 1 baseline.
* **Limitations:** Evaluation requires correlating the anomaly score back to $V_B$ anyway to validate physical relevance.

### Candidate D: Remaining Useful Life (RUL) Prediction
* **Target:** Number of remaining passes before $V_B \ge 0.3\text{ mm}$.
* **Feasibility:** **LOW / UNRECOMMENDED**.
* **Scientific Skepticism:**
  * Tool life varies wildly across cases (from 1 pass in Case 6 to 23 passes in Case 11).
  * 16 tools is far too small to establish an empirical distribution of RUL without making heavy artificial assumptions.
  * Forcing an RUL formulation here would feel synthetic and contrived — exactly what we are trying to fix from V1.

---

## 10. Technical Recommendation for Strategic Lead

1. **Primary Problem Formulation:**
   * **Problem A (Continuous $V_B$ Regression)** or **Problem B (Binary Tool Replacement Classification at $V_B = 0.30\text{ mm}$)**.
   * Both can even be modeled hierarchically: Regression predicts continuous $V_B$, with a calibrated decision boundary at $0.30\text{ mm}$ for the maintenance trigger.
2. **Handling Missing Values:**
   * Exclude the 21 `NaN` runs from supervised training/evaluation. (146 high-quality experimentally validated cuts is sufficient for disciplined modeling).
   * Monotonic linear interpolation across intermediate missing passes can be tested as a sensitivity check, but must not be mixed into the primary test benchmark.
3. **Handling Signal Disparity (Case 12 Run 1):**
   * Window all signals to the first 9,000 samples (36.0 seconds), or select the steady-state cutting zone (e.g. seconds 5 to 30) across all runs to ensure 100% temporal consistency.
4. **Validation Benchmark:**
   * **GroupKFold (Leave-One-Tool-Out)** grouped by `case`. Report Mean Absolute Error (MAE), RMSE, and $R^2$ across unseen tools.

---

## 11. Open Questions & Decision Points for Strategic Lead (ChatGPT)

1. **Problem Choice:** Does Strategic Lead prefer pure continuous regression ($V_B$), binary maintenance classification ($V_B \ge 0.30\text{ mm}$), or both?
2. **Material Stratification:** Should models be trained separately for Cast Iron vs Steel J45 (due to drastically different tool life and sensor physics), or as a single unified model conditioned on `material`?
3. **Missing Label Handling:** Strict drop of 21 unlabeled passes vs linear interpolation?

---

*Report prepared by AntiGravity Implementation Agent. No machine learning models have been trained. Awaiting Strategic Lead review.*
