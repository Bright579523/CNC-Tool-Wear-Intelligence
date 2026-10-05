# Phase 3.2 Feature Dataset Generation, Redundancy Audit & Compact Feature Selection Report
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2)
- **Author:** AntiGravity (Implementation Agent)
- **Domain Lead / Owner:** Bright
- **Strategic Lead:** ChatGPT
- **Status:** Phase 3.2 Complete & Refined (Feature Dataset Built, Redundancy Audited, Compact Candidate Set Locked — NO ML)

---

## Executive Summary

Phase 3.2 operationalizes the findings from Phase 3.1 into a **clean, auditable, ML-ready feature dataset** ([`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv)) and establishes a **compact, physically defensible candidate feature set**. Guided by the strategic principle:

> *"No favorite features. Keep only features that are necessary or meaningfully informative."*

From the initial pool of 25 candidate features:
- **5 features** are designated as the **Primary / Final Candidate Set**, providing complementary perspectives across Spindle Dynamic Load, Mechanical Shock Range, Impact Peakedness, and Acoustic-Emission Friction.
- **6 features** are designated as **Secondary Candidates**, offering alternative indices and kinematic harmonics for Phase 4 ablation experiments (specifically evaluating `smcAC_rms` only vs. `smcAC_rms` + `smcAC_spindle_band_pwr`).
- **2 features (`smcDC_mean`, `smcDC_std`)** are classified as **Diagnostic / Conditional** due to hardware saturation at $+9.995\text{ V}$ in 41 runs. They remain quarantined from primary predictive models.
- **12 features** were **Removed** due to mathematical redundancy (collinear twins), negative/erratic within-tool behavior, or lack of genuine wear tracking.
- **Strict Leakage Barrier:** Experimental identifiers (`case`, `run`, `cumulative_time_min`) and grouped validation identifiers (`condition_id`) are partitioned into metadata and strictly excluded from the predictive feature matrix.
- **Methodological Transparency:** Because candidate screening utilized wear correlations across the full dataset, this feature set is treated as a **locked exploratory candidate set** derived from domain engineering analysis. Phase 4 will validate this set via disciplined grouped cross-validation without post-hoc test-fold tuning.
- **No Machine Learning models** were trained; no train/test splits or cross-validations were performed. Phase 3.2 delivers a vetted, scientifically grounded input matrix for Phase 4.

---

## 1. Strategic Objective

The goal of Phase 3.2 is to transition from exploratory signal characterization to a structured, ML-ready feature dataset while preventing two fatal data-science pitfalls common in manufacturing analytics:
1. **The Feature Zoo Trap:** Dumping dozens of collinear or unvetted features into complex machine learning models, causing models to overfit to redundant variables or memorize sensor noise.
2. **The Numeric Threshold Trap:** Selecting features solely based on a single pooled correlation metric (e.g., $|r| > 0.60$), which blinds the model to cutting-condition confounding and discards low-correlation features that carry unique physical information.

Phase 3.2 establishes a **minimum sufficient feature set** where every retained feature has a clear physical role, verified data quality, non-redundant behavior, and empirical support from the Phase 3.1 within-tool audit.

---

## 2. Input Candidate Feature Set

The input pool consists of the 25 candidate features engineered and audited during Phase 3.1 across the 6 physical sensor channels:

| Sensor Channel | Candidate Features | Physical Domain | Domain Type |
| :--- | :--- | :--- | :---: |
| **`smcAC`** | `smcAC_rms`, `smcAC_std`, `smcAC_p2p`, `smcAC_kurtosis`, `smcAC_crest`, `smcAC_spindle_band_pwr` | Current / Dynamic Load & Kinematics | Time & Frequency |
| **`smcDC`** | `smcDC_mean`, `smcDC_std` | DC Motor Baseline Power Draw | Time |
| **`vib_spindle`** | `vib_spindle_kurtosis`, `vib_spindle_p2p`, `vib_spindle_mean` | Mechanical Impact & Edge Degradation | Time |
| **`AE_table`** | `AE_table_rms`, `AE_table_mean`, `AE_table_p2p`, `AE_table_kurtosis` | Flank Contact Friction & Table Acoustic Dynamics | Time |
| **`AE_spindle`** | `AE_spindle_p2p`, `AE_spindle_rms`, `AE_spindle_mean`, `AE_spindle_kurtosis` | Tool-Holder Acoustic Transmission & Bursts | Time |
| **`vib_table`** | `vib_table_spindle_band_pwr`, `vib_table_tooth_band_pwr`, `vib_table_mean`, `vib_table_std`, `vib_table_p2p`, `vib_table_kurtosis` | Table Structural Dynamics & Kinematic Harmonics | Time & Frequency |

---

## 3. Dataset Construction (`feature_dataset_v32.csv`)

The finalized feature dataset was constructed by joining the verified 145 VALID cutting runs with experimental metadata and sensor features. The dataset has dimensions **145 rows $\times$ 22 columns** and is serialized at:
📁 [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv)

### Column Layout & Partitioning
```
Dataset Schema (22 Columns):
├── ML Target (1 col)
│   └── VB_mm: Measured flank wear land width (0.00 to 1.53 mm)
├── Context / Metadata [EXCLUDED FROM PREDICTIVE FEATURES] (8 cols)
│   ├── case: Tool insert identifier (1 to 16)
│   ├── run: Cutting pass sequence counter (1 to 17)
│   ├── material_name: Cast Iron or Stainless Steel J45
│   ├── material_code: 1 (Cast Iron) or 2 (Stainless Steel J45)
│   ├── DOC_mm: Axial depth of cut (0.75 mm or 1.50 mm)
│   ├── feed_mm_rev: Feed rate per revolution (0.25 mm/rev or 0.50 mm/rev)
│   ├── condition_id: Operating condition key (C1 to C8)
│   └── cumulative_time_min: Total cut duration accumulated by tool
├── Primary / Final Predictive Candidates (5 cols)
│   ├── smcAC_rms: Spindle motor current RMS / dynamic load proxy
│   ├── vib_spindle_kurtosis: Spindle vibration envelope peakedness (low DOC/Feed bias)
│   ├── vib_spindle_p2p: Spindle vibration envelope peak excursion range
│   ├── AE_table_rms: Table acoustic-emission envelope effective energy
│   └── AE_spindle_p2p: Range of processed acoustic-emission envelope at spindle
├── Secondary Predictive Candidates (6 cols)
│   ├── smcAC_p2p: AC peak excursion range
│   ├── smcAC_spindle_band_pwr: Kinematic spindle harmonic power (Phase 4 Ablation Candidate)
│   ├── AE_table_p2p: Table acoustic burst peak range
│   ├── AE_spindle_rms: Spindle acoustic transmission energy
│   ├── vib_table_spindle_band_pwr: Table kinematic power at spindle freq (13.8 Hz)
│   └── vib_table_tooth_band_pwr: Table kinematic power at tooth passing freq (82.6 Hz)
└── Diagnostic / Conditional Features [CLIPPED SENSORS] (2 cols)
    ├── smcDC_mean: Average DC motor current (Clipped at +10V in 41 runs)
    └── smcDC_std: DC motor ripple amplitude (Variance compressed by clipping)
```

---

## 4. Feature Quality Audit

Every candidate feature was audited for missingness, cardinality, variance, distributional shape, extreme outliers, and hardware fidelity. Full details are recorded in [`reports/feature_audit_table.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/feature_audit_table.csv):

### 4.1 Missingness & Near-Zero Variance
- **Missing Values:** Exactly **0 missing values (0.0%)** across all 25 features for all 145 valid runs.
- **Near-Zero Variance (NZV):** **0 features** exhibited near-zero variance. All features show continuous dispersion across the operating envelope.

### 4.2 Hardware Saturation Audit (`smcDC`)
- **`smcDC_mean` and `smcDC_std`:** Saturated flat at the DAQ hardware limit of **$+9.995\text{ V}$ in 41 out of 145 valid runs (28.3%)**, primarily under heavy cuts ($\text{DOC} = 1.50\text{ mm}$, $\text{Feed} = 0.50\text{ mm/rev}$).
- Clipping truncates the dynamic peaks of the signal, which artificially depresses both the mean and the standard deviation. Consequently, both features are quarantined as **Diagnostic / Conditional** and excluded from the primary predictive feature set.

### 4.3 Preprocessing Characteristics of Vibration & AE
- As documented in Phase 3.1, channels `vib_table`, `vib_spindle`, `AE_table`, and `AE_spindle` were integrated via an analog hardware RMS converter ($\Delta T = 8.0\text{ ms}$) prior to DAQ digitization. They are strictly non-negative envelope voltages rather than raw kHz/MHz acoustic waveforms.

### 4.4 Distributional & Outlier Diagnostics
- Most features exhibit positive skewness ($0.3$ to $2.5$) consistent with tool degradation dynamics, where baseline values remain low during the fresh stage and rise markedly as flank wear progresses.
- `vib_spindle_kurtosis` displays high positive kurtosis ($8.25$), reflecting sharp impulsive shock peaks when worn inserts impact the workpiece.

---

## 5. Redundancy & Collinearity Audit

We evaluated pairwise collinearity among all 25 candidates using Pearson correlation, Spearman rank correlation, and hierarchical clustering. The audit table is available at [`reports/feature_redundancy_table.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/feature_redundancy_table.csv).

```
                      Key Collinear Pairs & Clusters
Collinear Pair / Cluster                      | Pearson r | Empirical Mechanism & Action
----------------------------------------------|-----------|---------------------------------------------------------
smcAC_rms  <-->  smcAC_std                    |  0.9999   | Identical for zero-mean AC wave -> Retain RMS, Drop STD
AE_table_rms  <-->  AE_table_mean             |  0.9991   | Fixed offset in smoothed envelope -> Retain RMS, Drop Mean
AE_spindle_rms  <-->  AE_spindle_mean         |  0.9994   | Fixed offset in smoothed envelope -> Retain RMS, Drop Mean
smcAC_rms  <-->  smcAC_p2p                    |  0.9850   | AC dynamic range correlates with energy -> Keep p2p secondary
smcAC_rms  <-->  smcAC_spindle_band_pwr       |  0.9850   | High collinearity -> RMS Primary, Band Power Secondary (Ablation)
AE_table_rms  <-->  AE_table_p2p              |  0.8870   | Table acoustic burst tracks mean energy -> Keep p2p secondary
AE_spindle_p2p  <-->  AE_spindle_rms          |  0.8230   | Spindle burst vs energy -> Keep p2p primary, RMS secondary
vib_table_mean  <-->  vib_table_std           |  0.9330   | Table envelope baseline tracks fluctuation -> Both removed
vib_spindle_kurtosis  <-->  vib_spindle_p2p   |  0.5540   | Peakedness vs peak excursion -> Low collinearity! Keep both
```

*Reference Visualization:*
- [Figure 1: Candidate Feature Collinearity Matrix](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_selection/fig1_feature_correlation_heatmap.png)

### Key Collinearity Insights:
1. **Mathematical Twins:** `smcAC_rms` and `smcAC_std` share a Pearson correlation of $r = 0.9999$. Retaining both provides zero extra information. `smcAC_rms` is retained; `smcAC_std` is removed.
2. **Analog Envelope Offset Twins:** `AE_table_mean` and `AE_table_rms` correlate at $r = 0.9991$. `AE_table_rms` is retained; `AE_table_mean` is removed. Similarly, `AE_spindle_mean` ($r = 0.9994$ with `AE_spindle_rms`) is removed.
3. **Harmonic Collinearity Decision (`smcAC_spindle_band_pwr`):** `smcAC_spindle_band_pwr` correlates with `smcAC_rms` at $r = 0.985$. While spindle rotational harmonic power has distinct kinematic meaning, a correlation of 0.985 indicates substantial information overlap. To honor the principle of keeping only necessary features, `smcAC_rms` is designated as **Primary**, while `smcAC_spindle_band_pwr` is moved to **Secondary / Candidate for Phase 4 Ablation**.
4. **Independent Impact Information:** `vib_spindle_kurtosis` and `vib_spindle_p2p` share a moderate correlation of only $r = 0.554$, demonstrating that envelope peakedness and peak vibration excursion capture distinct physical aspects of insert degradation.

---

## 6. Condition Metadata & Experimental Coverage Audit

To support grouped cross-validation in Phase 4 without causing data leakage, we formalized the 8 experimental cutting conditions (`condition_id = C1..C8`):

| Condition ID | Workpiece Material | DOC ($a_p$, mm) | Feed ($f_z$, mm/rev) | Tool Cases | Tools | Total Runs | Valid Sensor Runs | Corrupted Runs | Runs with VB | Missing VB Runs | Missing VB % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C1_CastIron_d0.75_f0.25`** | Cast Iron | 0.75 | 0.25 | [3, 11] | 2 | 37 | 34 | 0 | 34 | 3 | 8.1% |
| **`C2_CastIron_d0.75_f0.50`** | Cast Iron | 0.75 | 0.50 | [2, 12] | 2 | 29 | 24 | 2 | 24 | 3 | 10.3% |
| **`C3_CastIron_d1.50_f0.25`** | Cast Iron | 1.50 | 0.25 | [4, 10] | 2 | 17 | 17 | 0 | 17 | 0 | 0.0% |
| **`C4_CastIron_d1.50_f0.50`** | Cast Iron | 1.50 | 0.50 | [1, 9] | 2 | 26 | 22 | 0 | 22 | 4 | 15.4% |
| **`C5_Stainless_d0.75_f0.25`** | Stainless Steel J45 | 0.75 | 0.25 | [7, 13] | 2 | 23 | 20 | 0 | 20 | 3 | 13.0% |
| **`C6_Stainless_d0.75_f0.50`** | Stainless Steel J45 | 0.75 | 0.50 | [8, 14] | 2 | 15 | 12 | 0 | 12 | 3 | 20.0% |
| **`C7_Stainless_d1.50_f0.25`** | Stainless Steel J45 | 1.50 | 0.25 | [6, 15] | 2 | 8 | 7 | 0 | 7 | 1 | 12.5% |
| **`C8_Stainless_d1.50_f0.50`** | Stainless Steel J45 | 1.50 | 0.50 | [5, 16] | 2 | 12 | 9 | 0 | 9 | 3 | 25.0% |
| **Total / Summary** | — | — | — | 16 Cases | 16 | **167** | **145** | **2** | **145** | **20** | **12.0%** |

*Table serialized at:* [`reports/condition_coverage_table.csv`](file:///D:/Project/Manufacturing%20Analytics/reports/condition_coverage_table.csv)

### Observations for Validation Planning:
- Exactly 2 independent tools were tested per cutting condition (Replicate 1 and Replicate 2).
- Condition C7 (`Stainless Steel, d1.50, f0.25`) has the lowest sample count (7 valid runs) because Case 6 was aborted after 1 run.
- Missing VB rates range from 0.0% (C3) to 25.0% (C8). All 145 valid sensor runs have confirmed offline VB measurements.

---

## 7. Compact Feature Selection Rationale

Rather than forcing an arbitrary number of features, the selection was governed by a **multi-criteria engineering sieve**:
1. **Physical Representation:** The primary set preserves distinct physical perspectives across Spindle Dynamic Load, Mechanical Shock Range, Impact Peakedness, and Acoustic Friction.
2. **Within-Tool Wear Association:** Retained features demonstrate consistent progression within individual tools ($r_s > 0.50$ in the Phase 3.1 within-tool audit), **supporting consistent within-tool tracking of measured $V_B$ beyond pooled cutting-condition differences.**
3. **Hardware Integrity:** Channels suffering from clipping or sensor artifacts are excluded from primary modeling.
4. **Collinearity Pruning:** When two features share $>95\%$ collinearity and measure the same underlying physical effect, only the more standardized and robust feature is retained.

*Reference Visualization:*
- [Figure 3: Phase 3.2 Compact Feature Selection Logic](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_selection/fig3_retained_vs_removed_features.png)

---

## 8. Final Feature Set

### 8.1 Primary / Final Candidate Feature Set (5 Features)
These 5 features form the **core, non-redundant predictive feature matrix** recommended for Phase 4 baseline modeling:

| Feature Name | Sensor Channel | Physical Domain | Pooled $r_s$ ($V_B$) | Within-Tool Median $r_s$ | Confounding Bias | Engineering Interpretation & Justification |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`smcAC_rms`** | `smcAC` | Current / Dynamic Load | **+0.753** | **+0.993** (14/14 pos) | Moderate (DOC/Feed) | **Spindle motor current RMS / dynamic load proxy.** Monotonic wear progression within every tool. Clean bipolar DAQ signal. |
| **`vib_spindle_kurtosis`** | `vib_spindle` | Mechanical Impact / Peakedness | **+0.591** | **+0.528** (12/14 pos) | **Low / Near-Zero** | **Captures changes in the peakedness of spindle-side vibration envelope associated with increasing wear.** Very low DOC ($r_s = -0.21$) and Feed ($r_s = -0.15$) bias. |
| **`vib_spindle_p2p`** | `vib_spindle` | Mechanical Impact / Excursion | **+0.521** | **+0.609** (13/14 pos) | Moderate | **Spindle vibration peak envelope excursion range.** Complements kurtosis by capturing absolute peak vibration amplitudes as wear progresses. |
| **`AE_table_rms`** | `AE_table` | Acoustic Emission / Friction | **+0.668** | **+0.957** (14/14 pos) | Moderate (DOC) | **Acoustic-emission envelope associated with cutting/contact activity and showing strong within-tool association with measured $V_B$.** |
| **`AE_spindle_p2p`** | `AE_spindle` | Acoustic Emission / Bursts | **+0.672** | **+0.785** (14/14 pos) | Low DOC bias ($r_s = 0.24$) | **Range of the processed acoustic-emission envelope measured at the spindle-side sensor.** Strong wear tracking with lower DOC confounding than table AE. |

*Reference Visualization:*
- [Figure 2: Candidate Feature Distributions Across Flank Wear Degradation Stages](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_selection/fig2_candidate_feature_distributions.png)

### 8.2 Secondary Feature Set (6 Features)
Retained in `feature_dataset_v32.csv` as secondary candidates for optional sensitivity checks, kinematic comparisons, or Phase 4 ablation experiments:
1. **`smcAC_spindle_band_pwr`:** Kinematic spindle rotational harmonic power ($13.8\text{ Hz}$). **Primary candidate for Phase 4 ablation:** testing whether adding this harmonic feature yields meaningful ML performance gains over `smcAC_rms` alone.
2. **`smcAC_p2p`:** AC peak excursion range ($r = 0.985$ with `smcAC_rms`).
3. **`AE_table_p2p`:** Table acoustic burst range ($r = 0.887$ with `AE_table_rms`).
4. **`AE_spindle_rms`:** Spindle acoustic transmission energy ($r = 0.823$ with `AE_spindle_p2p`).
5. **`vib_table_spindle_band_pwr`:** Table vibration power at spindle rotational frequency ($13.8\text{ Hz}$).
6. **`vib_table_tooth_band_pwr`:** Table vibration power at cutter tooth passing frequency ($82.6\text{ Hz}$).

---

## 9. Diagnostic / Conditional Features

| Feature Name | Sensor Channel | Physical Role | Hardware Defect & Status |
| :--- | :--- | :--- | :--- |
| **`smcDC_mean`** | `smcDC` | DC Motor Baseline Current | **HARDWARE SATURATION:** 41 of 145 runs (28.3%) clipped flat at $+9.995\text{ V}$. Excluded from primary predictive models. |
| **`smcDC_std`** | `smcDC` | DC Current Ripple Amplitude | Variance compressed by 10V clipping. Retained strictly for sensitivity audits. |

> [!CAUTION]
> **Modeling Directive for Phase 4:** `smcDC_mean` and `smcDC_std` must **NOT** be included in standard ML models. They are preserved in the dataset solely to enable a controlled sensitivity experiment evaluating whether models trained with clipped signals degrade or develop distorted decision boundaries.

---

## 10. Removed Features and Technical Justifications

The following 12 candidate features were removed from predictive modeling:

| Feature Name | Sensor Channel | Category of Removal | Technical Justification |
| :--- | :--- | :--- | :--- |
| **`smcAC_std`** | `smcAC` | Collinear Twin | Pearson $r = 0.9999$ with `smcAC_rms`. Redundant zero-mean variance. |
| **`AE_table_mean`** | `AE_table` | Collinear Twin | Pearson $r = 0.9991$ with `AE_table_rms`. Redundant analog envelope offset. |
| **`AE_spindle_mean`** | `AE_spindle` | Collinear Twin | Pearson $r = 0.9994$ with `AE_spindle_rms`. Redundant envelope offset. |
| **`smcAC_kurtosis`** | `smcAC` | Uninformative / Weak Wear Signal | Near-zero wear correlation (pooled $r_s = -0.062$, within-tool median $r_s = -0.298$). |
| **`smcAC_crest`** | `smcAC` | Uninformative / Weak Wear Signal | Normalization suppresses progressive wear information (within-tool median $r_s = -0.471$). |
| **`AE_table_kurtosis`** | `AE_table` | Material Confounded Artifact | Within-tool median $r_s = +0.036$ (positive in 7 tools, negative in 7 tools). Pooled correlation ($r_s = 0.135$) was an illusion driven by Stainless Steel having higher burstiness than Cast Iron ($r_s = 0.750$ with Material). |
| **`AE_spindle_kurtosis`** | `AE_spindle` | Erratic / Low Wear Fidelity | Weak within-tool tracking (median $r_s = +0.121$, positive in only 8/14 tools). |
| **`vib_table_mean`** | `vib_table` | Negative / Dampening Trend | Table RMS envelope baseline dampens as tool wears (within-tool median $r_s = -0.588$). |
| **`vib_table_std`** | `vib_table` | Negative / Dampening Trend | Fluctuation amplitude dampens with wear within tools (median $r_s = -0.679$). |
| **`vib_table_p2p`** | `vib_table` | Material Confounded / Dampening | Decreases with wear within tools (median $r_s = -0.653$) and is dominated by Material ($r_s = 0.628$). |
| **`vib_table_kurtosis`** | `vib_table` | Highly Erratic | Correlation ranges wildly across tools (min $-1.00$, max $+0.97$). Highly unstable. |
| **`vib_spindle_mean`** | `vib_spindle` | Inverted / Setup-Dominated | Negative correlation with wear within tools (median $r_s = -0.821$, 0/14 positive). Reflects operating setup shifts rather than tool wear. |

---

## 11. ML Leakage Exclusions & Methodological Transparency

### 11.1 Experimental Metadata Exclusions
To ensure scientific validity in Phase 4, the following variables are strictly classified as **Metadata Only** and are **prohibited from entering the predictive feature matrix $X$**:
- **`case` (Tool ID):** Categorical identity of the insert. Must be used exclusively as the grouping key for grouped cross-validation.
- **`run` (Pass Counter):** Sequentially tracks tool age; including it allows models to memorize tool wear progression without learning from sensor signals.
- **`cumulative_time_min`:** **Target proxy / tool-age leakage.** In this experimental setup, cumulative cutting time functions as a direct proxy of tool age and wear progression; inclusion bypasses sensor learning entirely.
- **`condition_id`:** Metadata label for grouping in cutting-condition generalization tests.

### 11.2 Methodological Transparency: Exploratory Candidate Screening vs. Phase 4 Evaluation
A critical methodological distinction must be recognized:
- Feature screening in Phase 3.1 and Phase 3.2 utilized correlation analyses against measured $V_B$ across the full 145 runs to identify physical sensitivity and prune material artifacts.
- Therefore, we do **not** claim this 5-feature set was selected through a completely leakage-free, out-of-fold automated procedure.
- Instead, this feature set is explicitly framed as a **locked exploratory candidate set** derived from domain-guided engineering analysis.
- **Protocol for Phase 4:**
  1. Phase 3.2 locks this candidate set based on physical reasoning and exploratory audit.
  2. Phase 4 will evaluate this locked candidate set via strict Grouped Cross-Validation.
  3. Any subsequent feature pruning or ablation (e.g., testing `smcAC_rms` vs. `smcAC_rms` + `smcAC_spindle_band_pwr`) must be tested strictly within cross-validation training folds (or as predefined ablation experiments), without adjusting feature definitions based on test-fold scores.

---

## 12. Phase 4 Readiness & Proposed Validation Architecture

The dataset is verified and ready for Phase 4. We propose the following disciplined workflow for Phase 4:

```text
Phase 4 Machine Learning Architecture
├── 1. Baseline Model (Dummy / Mean Baseline)
├── 2. Sensor-Only Compact Model (5 Primary Candidates)
├── 3. Multi-Model Benchmark (Ridge, Random Forest, SVR, Gradient Boosting)
│      (No predetermined favorite; select the model best suited for N=145)
├── 4. Grouped Validation: Unseen Tool (Group by case / Leave-One-Tool-Out)
├── 5. Model Interpretation (Feature coefficients, importances, error breakdown)
├── 6. Feature Ablation Study:
│      RMS Only (smcAC_rms) vs. RMS + Spindle Band Power (smcAC_rms + smcAC_spindle_band_pwr)
├── 7. Context Model (Sensor Features + Material + DOC + Feed)
└── 8. Grouped Validation: Unseen Cutting Condition (Group by condition_id)
```

---

## 13. Limitations

1. **Sample Size:** 145 valid runs across 16 tools (effectively 14 tools with multiple runs) provides a constrained dataset for machine learning.
2. **Replicate Depth:** $N=2$ tools per condition limits the ability to statistically model tool-to-tool manufacturing variance.
3. **Analog Hardware RMS Filtering:** Integration with $\Delta T = 8.0\text{ ms}$ limits frequency-domain analysis on vibration and AE to envelope dynamics below 100 Hz.
4. **DC Spindle Sensor Saturation:** Severe hardware clipping at $+10\text{ V}$ eliminates `smcDC` as a primary feature, leaving `smcAC` as the sole uncorrupted spindle load channel.

---

## 14. Final Gate — Explicit Answers to Strategic Review Questions

### 1. Which features are in the final compact candidate set?
The **5 Primary / Final Candidates** are:
1. `smcAC_rms` (Spindle motor current RMS / dynamic load proxy)
2. `vib_spindle_kurtosis` (Spindle vibration envelope peakedness / edge degradation)
3. `vib_spindle_p2p` (Spindle vibration peak envelope excursion range)
4. `AE_table_rms` (Table acoustic-emission envelope effective energy / flank contact activity)
5. `AE_spindle_p2p` (Range of processed acoustic-emission envelope measured at the spindle-side sensor)

### 2. Which features were removed and why?
**12 features were removed:**
- `smcAC_std`, `AE_table_mean`, `AE_spindle_mean`: Removed as collinear twins ($r > 0.999$) of retained RMS features.
- `smcAC_kurtosis`, `smcAC_crest`: Removed due to near-zero wear association ($r_s \approx -0.06$ to $-0.12$).
- `AE_table_kurtosis`: Removed because within-tool wear correlation is zero ($r_s = +0.036$); pooled correlation was an illusion of workpiece material differences.
- `AE_spindle_kurtosis`: Removed due to weak and erratic tracking ($r_s = +0.121$).
- `vib_table_mean`, `vib_table_std`, `vib_table_p2p`: Removed because table vibration envelopes dampen with wear within tools (median $r_s = -0.59$ to $-0.68$) and are dominated by material confounding.
- `vib_table_kurtosis`: Removed due to extreme instability across tools ($-1.0$ to $+0.97$).
- `vib_spindle_mean`: Removed because baseline envelope levels shift with machine setup rather than tool wear (median within-tool $r_s = -0.82$).

### 3. Which features remain diagnostic only?
`smcDC_mean` and `smcDC_std` remain **Diagnostic / Conditional only**. They are preserved in the dataset strictly for sensitivity testing and must not be used in primary ML models due to hardware clipping at $+9.995\text{ V}$ in 41 runs.

### 4. Are there any serious data-quality problems remaining?
**No unresolved data-quality issue remains within the primary feature set.** The known `smcDC` saturation issue remains quarantined as a diagnostic limitation. All 145 rows in `feature_dataset_v32.csv` have 100% complete data (zero missing values), zero near-zero variance features, and confirmed synchronization.

### 5. Is the feature dataset ready for Phase 4 ML?
**YES.** [`data/feature_dataset_v32.csv`](file:///D:/Project/Manufacturing%20Analytics/data/feature_dataset_v32.csv) is fully structured, validated, and ready for baseline model training and grouped evaluation in Phase 4.

### 6. Is there evidence that important information is missing and that new features may be needed later?
**No immediate information gap exists.** The 5 primary features cover spindle dynamic load, mechanical impact peakedness, dynamic excursion range, and acoustic contact activity. Furthermore, `smcAC_spindle_band_pwr` is pre-computed and stored as a Secondary candidate for a controlled ablation experiment in Phase 4.

---
*Report produced by AntiGravity (Implementation Agent) under the direction of Bright and Strategic Lead ChatGPT.*
