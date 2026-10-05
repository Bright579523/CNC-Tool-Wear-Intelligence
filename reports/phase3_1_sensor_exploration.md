# Phase 3.1 Sensor Signal Exploration & Feature Engineering Design Report
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2)
- **Author:** AntiGravity (Implementation Agent)
- **Domain Lead / Owner:** Bright
- **Strategic Lead:** ChatGPT
- **Status:** Phase 3.1 Complete & Refined (Exploration, DAQ Audit & Physical Feature Design — NO ML)

---

## Executive Overview

Phase 3.1 establishes the **physical signal exploration, data acquisition (DAQ) quality audit, within-tool wear sanity check, and domain-grounded feature design** across the 6 sensor channels in the NASA Milling dataset. In accordance with the project's **manufacturing-first, data-science-supported** ethos:
- **No Machine Learning models** (Random Forest, XGBoost, Ridge, SVR) were trained.
- **No synthetic feature zoo** was constructed (no unmotivated wavelets or blind high-dimensional decompositions).
- **Tool-age proxies** (`run` number, `cumulative_time_min`, `case` ID) are strictly audited and excluded from sensor feature sets to prevent fatal data leakage.
- All 145 VALID cutting runs were analyzed to understand sensor physics, audit hardware artifacts (clipping/saturation), evaluate kinematic frequency content, test windowing stability, and separate genuine tool wear sensitivity from cutting condition confounding (Material, Feed, DOC).
- The candidate feature set is structured into **purpose-driven physical domains** (Load, Mechanical Impact, Friction, Kinematics) and screened via a **within-tool sanity check** to prepare for redundancy pruning in Phase 3.2.

---

## 1. Physical Understanding of the Sensor Signals

### 1.1 Sensor Channels & Preprocessing Architecture
According to the experimental documentation (Goebel & Agogino, 1999; *Readme.pdf*, pp. 2–5) and literature utilizing this benchmark, the 6 channels are acquired via a National Instruments MIO-16 board installed in an IBM PC 486DX:

| Channel | Sensor Hardware | Physical Quantity | Preprocessing in Hardware | Verified Signal Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`smcAC`** | CTA 213 Current Sensor | AC Spindle Motor Current | Amplified only (Direct to DAQ) | Symmetrical bipolar oscillation ($\pm 7\text{ V}$), zero-centered, reflects dynamic cutting torque harmonics. |
| **`smcDC`** | OMRON K3TB-A1015 Converter | DC Spindle Motor Current | Amplified only (Direct to DAQ) | Unipolar positive signal ($0 - 10\text{ V}$), reflects baseline cutting power draw. **Saturates at +10V in heavy cuts.** |
| **`vib_table`** | ENDEVCO 7201-50 Accelerometer | Table Vibration | High-pass + 400 Hz Low-pass + **Analog RMS Device ($\Delta T = 8.0\text{ ms}$)** | **Strictly non-negative voltage ($0 - 3.7\text{ V}$)** representing the 8 ms smoothed RMS vibration envelope. |
| **`vib_spindle`** | ENDEVCO 7201-50 Accelerometer | Spindle Vibration | High-pass + 400 Hz Low-pass + **Analog RMS Device ($\Delta T = 8.0\text{ ms}$)** | Strictly non-negative voltage ($0.2 - 2.5\text{ V}$), stable baseline with sharp transient impact peaks. |
| **`AE_table`** | Physical Acoustics WD 925 | Table Acoustic Emission | 50 kHz Preamplifier + High-pass + **Analog RMS Device ($\Delta T = 8.0\text{ ms}$)** | Strictly non-negative envelope voltage ($0.01 - 1.1\text{ V}$), captures continuous flank friction/deformation energy. |
| **`AE_spindle`** | Physical Acoustics WD 925 | Spindle Acoustic Emission | 50 kHz Preamplifier + High-pass + **Analog RMS Device ($\Delta T = 8.0\text{ ms}$)** | Strictly non-negative envelope voltage ($0.01 - 1.25\text{ V}$), captures tool-holder/bearing acoustic emissions. |

> [!IMPORTANT]
> **Domain Clarification on Preprocessing:**  
> The signals stored in `vib_table`, `vib_spindle`, `AE_table`, and `AE_spindle` are **not raw high-frequency waveforms (not raw MHz acoustic emission or kHz structural accelerations)**. They were passed through an analog hardware RMS converter with an integration time constant $\Delta T = 8.00\text{ ms}$ prior to analog-to-digital conversion. Consequently, these channels represent **smoothed RMS envelope voltages**, explaining why their digitized values are strictly non-negative and capture low-frequency envelope modulations rather than high-frequency stress waves.

*Reference Visualizations:*
- [Figure 1: Raw Signals Fresh vs. Worn (Cast Iron)](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_signal_examples/fig1_raw_signals_fresh_vs_worn_cast_iron.png)
- [Figure 2: Raw Signals Fresh vs. Worn (Stainless Steel J45)](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_signal_examples/fig2_raw_signals_fresh_vs_worn_stainless_steel.png)

---

## 2. Signal-Quality Audit & Hardware Artifacts

### 2.1 Critical Finding: `smcDC` Clipping/Saturation at +10.0 V
- **Observation:** In **41 out of 145 VALID runs (28.3%)**, the DC motor current signal (`smcDC`) saturates flat at **$+9.995\text{ V}$**, reaching the absolute hardware input limit of the DAQ board.
- **Affected Conditions:** Primarily Case 1, Case 4, Case 5, Case 9, Case 10, and Case 16 — all operating under heavy cutting parameters ($\text{DOC} = 1.50\text{ mm}$ and/or $\text{Feed} = 0.50\text{ mm/rev}$). In Case 1 Run 10, **62.8% of all samples are clipped flat at 9.995 V**.
- **Impact on Statistics:** Hardware clipping truncates the upper dynamic peaks of the signal. This distorts not only the mean (`smcDC_mean`) but also compresses the variance and standard deviation (`smcDC_std`).
- **Methodological Status:** While `smcDC` shows a high pooled correlation with $V_B$ ($r_s = 0.783$), this channel carries significant data-quality risk. Both `smcDC_mean` and `smcDC_std` are categorized as **`Diagnostic / Conditional`**. They must **NOT** be included as primary features in Phase 4 ML until we evaluate whether saturation introduces severe non-linear prediction artifacts.
- None of the other five channels (`smcAC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`) exhibit clipping or saturation across the 145 valid runs.

### 2.2 DAQ Synchronization and Sampling Rigor
- **Sampling Frequency:** Exactly **$f_s = 250.0\text{ Hz}$** across all channels.
- **Record Length:** Exactly **9,000 samples per channel** across all 145 VALID runs, yielding an exact snapshot duration of:
  $$T_{\text{snapshot}} = \frac{9,000}{250.0} = 36.00\text{ seconds}$$
- **Synchronization:** All 6 channels were digitized simultaneously via the sample-and-hold MIO-16 board.

*Reference Visualization:*
- [Figure 3: DC Motor Current (smcDC) Saturation and Clipping](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_signal_examples/fig3_signal_saturation_and_clipping_smcDC.png)

---

## 3. Time-Domain Signal Dynamics

Time-domain exploratory statistics across all 145 valid runs reveal distinct physical relationships:

1. **RMS and Standard Deviation (`smcAC`, `AE_table`, `AE_spindle`):**
   - For zero-mean signals like `smcAC`, RMS and standard deviation are mathematically near-identical ($r = 0.9999$), indicating redundant information.
   - `smcAC_rms` exhibits a strong, monotonic progression as tools wear (e.g., in Case 3, increasing from $0.87\text{ V}$ on fresh inserts to $1.89\text{ V}$ on worn inserts, a $+117\%$ increase). As the flank wear land expands, friction and tangential cutting forces increase, demanding higher spindle motor current.
   - `AE_table_rms` increases consistently with wear (e.g., in Case 3, from $0.139\text{ V}$ to $0.245\text{ V}$, a $+76\%$ increase), capturing the continuous acoustic emission generated by the rubbing contact between the flank wear land and the cut workpiece surface.
2. **Peak-to-Peak Amplitude (`p2p`):**
   - In `vib_spindle_p2p` and `AE_spindle_p2p`, peak dynamic excursions increase markedly with advanced cutting-edge degradation ($r_s = 0.52 - 0.67$ with $V_B$).
3. **Kurtosis & Crest Factor:**
   - `vib_spindle_kurtosis` increases as wear progresses ($r_s = 0.591$ with $V_B$), reflecting impulsive shock peaks superimposed on a quiet baseline when degraded inserts impact the workpiece.
   - Conversely, `AE_table_kurtosis` correlates poorly with $V_B$ ($r_s = 0.135$), but correlates strongly with Workpiece Material ($r_s = 0.750$). Stainless Steel J45 produces high-kurtosis acoustic bursts, whereas Cast Iron produces a continuous, Gaussian-like acoustic background.

---

## 4. Frequency-Domain Exploration & Kinematic Alignment

Fast Fourier Transform (FFT) analysis on detrended signals was performed to assess alignment with machine kinematics:

### 4.1 Kinematic Cutting Frequencies
- Spindle Rotational Frequency:
  $$f_{\text{spindle}} = \frac{n}{60} = \frac{826\text{ RPM}}{60} = 13.767\text{ Hz} \approx 13.77\text{ Hz}$$
- Cutter Tooth Passing Frequency (TPF) for 6 inserts:
  $$f_{\text{tooth}} = 6 \times f_{\text{spindle}} = 6 \times 13.767 = 82.60\text{ Hz}$$
- Nyquist Limit:
  $$f_{\text{Nyquist}} = \frac{250\text{ Hz}}{2} = 125.0\text{ Hz}$$

### 4.2 Observed Spectral Behavior
- **Spindle Rotational Peak (13.77 Hz):**
  - In `vib_table`, `AE_table`, and `smcAC`, distinct spectral peaks appear consistently at **$13.72 - 13.77\text{ Hz}$** across cutting runs.
  - *Engineering Interpretation:* Spectral components aligned with spindle rotational and tooth-passing frequencies were observed in the processed sensor envelopes. In `smcAC`, dynamic power at $13.8\text{ Hz}$ (`smcAC_spindle_band_pwr`) tracks wear progression with a rank correlation of **$r_s = 0.756$**.
- **Tooth Passing Frequency Peak (82.60 Hz):**
  - In `vib_table`, a secondary harmonic emerges at **$82.6\text{ Hz}$**, capturing individual tooth engagements.
  - Because the analog hardware RMS converter integrated the raw high-frequency signals with $\Delta T = 8.0\text{ ms}$ (imposing an effective low-pass cutoff $\sim 125\text{ Hz}$), high-frequency cutting harmonics above 100 Hz are significantly attenuated.
- **Frequency Domain Feature Recommendation:**
  - **Justified:** Concentrated band powers around known kinematics:
    1. Spindle Rotational Band: $11.0 - 16.5\text{ Hz}$
    2. Tooth Passing Band: $75.0 - 90.0\text{ Hz}$
  - **Not Justified:** Blind arbitrary frequency binning (e.g., 20 uniform bins) or high-frequency wavelet sub-bands, because the $250\text{ Hz}$ sampling rate and analog RMS filtering restrict the meaningful dynamic spectrum to $<100\text{ Hz}$.

*Reference Visualization:*
- [Figure 4: Frequency Spectra & Kinematic Harmonics](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_frequency_analysis/fig4_fft_spectra_and_cutting_harmonics.png)

---

## 5. Windowing Findings: Full-Window vs. Sub-Window Analysis

We evaluated whether segmenting the 36-second recording into sub-windows is necessary:

1. **Collinearity Test (Full-Window vs. Steady-State Window):**
   - Comparing the Full-Window RMS ($0 - 36\text{ s}$, 9,000 samples) against a pure Steady-State Mid-Window ($10 - 30\text{ s}$, 5,000 samples) yielded a Pearson correlation of **$r = 0.9960$** for `smcAC`, **$r = 0.9984$** for `vib_table`, and **$r = 0.9991$** for `AE_table`.
   - The full-window feature captures practically identical wear and condition information as the middle-window feature.
2. **Within-Pass Transient Dynamics:**
   - In the first 5–8 seconds of each run, a mild entry transient occurs as the $\varnothing 70\text{ mm}$ cutter engages the workpiece.
   - The ratio of late-window RMS ($24-36\text{ s}$) to early-window RMS ($0-12\text{ s}$) averages $1.15 - 1.30$, reflecting this entry stabilization rather than rapid within-pass wear.
3. **Windowing Recommendation:**
   - **Primary Approach:** Use **Full-Window features ($0 - 36\text{ s}$)** as the baseline. They maximize statistical stability (9,000 samples) and avoid arbitrary boundary selection.
   - **Secondary Diagnostic:** Sub-window drift ratios provide marginal extra information ($r < 0.20$ with $V_B$) and add unnecessary dimensionality. Sub-windowing is therefore **NOT recommended** for the primary feature set.

*Reference Visualization:*
- [Figure 5: Windowing Analysis — Full vs. Sub-Window Evaluation](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_frequency_analysis/fig5_windowing_full_vs_subwindow_analysis.png)

---

## 6. Relationship to Measured Flank Wear ($V_B$) & Sanity Checks

### 6.1 Pooled Correlation Audit: Correlation $\neq$ Wear Sensitivity
A critical methodological trap in machining analytics is treating pooled correlation across all 145 runs as direct proof of "wear sensitivity." In this dataset:
$$\text{Cutting Run} \longrightarrow \{\text{Material, DOC, Feed, Tool Age}\} \longrightarrow \text{Sensors} \longleftrightarrow V_B$$

A high pooled correlation can arise simply because a feature responds strongly to heavy cutting conditions (e.g., $\text{DOC} = 1.50\text{ mm}$) where wear also happens to accumulate faster.

```
                        Spearman Correlation Matrix (Selected Candidates)
                     Target                          Confounders
Feature            | Spearman_VB | Spearman_DOC | Spearman_Feed | Spearman_Material | Risk Assessment
-------------------|-------------|--------------|---------------|-------------------|----------------------------------
smcDC_mean         |   +0.783    |    +0.523    |    +0.311     |      +0.172       | HIGH RISK (41 Runs Clipped at 10V)
smcAC_spindle_pwr  |   +0.756    |    +0.550    |    +0.400     |      +0.147       | Wear Association + Load Confounded
smcAC_rms          |   +0.753    |    +0.553    |    +0.399     |      +0.148       | Wear Association + Load Confounded
smcAC_p2p          |   +0.685    |    +0.583    |    +0.474     |      +0.138       | Wear Association + Load Confounded
AE_table_rms       |   +0.668    |    +0.537    |    +0.325     |      -0.237       | Strong Friction/Wear Association
AE_spindle_p2p     |   +0.672    |    +0.237    |    +0.348     |      -0.120       | Wear Association, Lower DOC Bias
vib_spindle_kurt   |   +0.591    |    -0.214    |    -0.150     |      +0.203       | Impact Wear Sensitive (Low Bias!)
vib_table_p2p      |   -0.231    |    +0.253    |    +0.224     |      +0.628       | Material-Dominated (Weak Wear)
AE_table_kurtosis  |   +0.135    |    -0.208    |    -0.395     |      +0.750       | Material-Dominated (Weak Wear)
```

> [!NOTE]
> **Wording Precision:** Rather than claiming `smcAC_rms` is an unconfounded wear indicator, the empirical evidence demonstrates that **`smcAC_rms` shows a strong positive association with measured $V_B$, but the relationship is also heavily influenced by cutting conditions (DOC and Feed).**

---

### 6.2 Within-Tool / Within-Condition Sanity Check
To decouple cutting condition confounding from true tool degradation, we conducted a **Within-Tool Sanity Check**. For each candidate feature, the Spearman rank correlation with $V_B$ was computed **strictly inside each individual tool (Case)**, where Material, DOC, and Feed are 100% constant. We then evaluated the median within-tool correlation across all 14 evaluated tool inserts (excluding Case 6 which aborted after 1 run):

| Feature Name | Sensor Channel | Physical Purpose | Pooled Spearman $r_s$ (145 Runs) | Within-Tool Median Spearman $r_s$ | Tools with $r_s > 0$ | Tools with $r_s > 0.50$ | Sanity Check Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`smcAC_rms`** | Current | Spindle Dynamic Torque | **+0.753** | **+0.993** | **14 / 14** | **14 / 14** | 🟢 **PASS:** Monotonic wear growth within EVERY tool (all $r_s > 0.94$). |
| **`smcAC_spindle_band_pwr`** | Frequency | Kinematic Spindle Harmonic | **+0.756** | **+0.993** | **14 / 14** | **14 / 14** | 🟢 **PASS:** Near-perfect wear tracking across all tools. |
| **`AE_table_rms`** | Acoustic Emission | Flank Friction Energy | **+0.668** | **+0.957** | **14 / 14** | **14 / 14** | 🟢 **PASS:** Flank rubbing energy increases in all 14 tools (all $r_s > 0.64$). |
| **`AE_spindle_p2p`** | Acoustic Emission | Tool Holder Micro-bursts | **+0.672** | **+0.785** | **14 / 14** | **14 / 14** | 🟢 **PASS:** Consistent positive wear tracking across all inserts. |
| **`vib_spindle_p2p`** | Vibration | Spindle Mechanical Shock | **+0.521** | **+0.609** | **13 / 14** | **8 / 14** | 🟢 **PASS:** Positive tracking with low cutting condition bias. |
| **`vib_spindle_kurtosis`** | Vibration | Edge Degradation Peakedness | **+0.591** | **+0.528** | **12 / 14** | **7 / 14** | 🟢 **PASS:** High wear specificity, uncoupled from DOC/Feed. |
| **`smcDC_mean`** | Current | DC Baseline Power | **+0.783** | **+0.983** | **14 / 14** | **14 / 14** | 🟡 **CONDITIONAL:** High correlation, but 41 runs clipped flat at +10V. |
| **`smcDC_std`** | Current | DC Current Ripple | **+0.596** | **+0.957** | **14 / 14** | **12 / 14** | 🟡 **CONDITIONAL:** Variance distorted by hardware clipping. |
| **`AE_table_kurtosis`** | Acoustic Emission | Acoustic Peakedness | **+0.135** | **+0.036** | **7 / 14** | **2 / 14** | 🔴 **FAIL:** Zero wear tracking (random sign); pooled $r_s$ was a material artifact. |
| **`vib_table_mean`** | Vibration | Table Vibration Baseline | **-0.205** | **-0.588** | **5 / 14** | **2 / 14** | 🔴 **FAIL:** Negative trend; table envelope dampens with wear. |
| **`vib_table_p2p`** | Vibration | Table Envelope Range | **-0.231** | **-0.653** | **2 / 14** | **0 / 14** | 🔴 **FAIL:** Reflects material metallurgy rather than wear. |

### Key Insights from Within-Tool Sanity Check
1. **Confirmation of Genuine Wear Signal:** Features like `smcAC_rms` and `AE_table_rms` achieve median within-tool correlations $>0.95$ across 100% of tested tools. This proves their high pooled correlation is not merely an artifact of "knowing which case is running" — they genuinely track progressive wear within constant cutting conditions.
2. **Exposure of Material Artifacts:** `AE_table_kurtosis` had a slight positive pooled correlation ($+0.135$), but its within-tool median is $+0.036$ (positive in 7 tools, negative in 7 tools). This proves that `AE_table_kurtosis` has no wear tracking capability; its pooled correlation was entirely an illusion driven by Stainless Steel having higher kurtosis than Cast Iron.
3. **Table Vibration Damping:** Table vibration features (`vib_table_mean`, `vib_table_p2p`) correlate negatively with wear within tools (median $r_s = -0.59$ to $-0.65$), suggesting the table vibration envelope dampens as the insert rubs.

*Reference Visualizations:*
- [Figure 6: Candidate Features vs. VB Stratified by Cutting Conditions](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_distributions/fig6_features_vs_VB_with_confounding.png)
- [Figure 7: Feature Correlation Audit Heatmap](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_distributions/fig7_feature_correlation_heatmap.png)
- [Figure 9: Within-Tool vs. Pooled Wear Correlation Sanity Check](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_distributions/fig9_within_tool_vs_pooled_correlation.png)

---

## 7. Replicate Repeatability Audit & ML Validation Strategy

### 7.1 Repeatability Across Independent Inserts (N=2 Tools)
Comparing feature trajectories between Replicate 1 (Cases 1–8) and Replicate 2 (Cases 9–16) under matched cutting conditions:
- **Condition 1 (Cast Iron, d0.75 f0.25):** Case 3 and Case 11 show closely matched `smcAC_rms` trajectories, starting at $0.85 - 0.90\text{ V}$ and climbing smoothly to $1.40 - 1.85\text{ V}$ as tools degrade.
- **Condition 5 (Stainless Steel J45, d0.75 f0.25):** Case 7 and Case 13 exhibit similar baseline levels ($0.88 - 0.95\text{ V}$) and identical upward trends.
- **Methodological Assessment:** The replicate trajectories provide descriptive support for evaluating grouped validation across tool IDs; however, the small number of independent tools ($N=2$ per condition) limits conclusions about generalization.

### 7.2 Two Distinct Validation Paradigms for Phase 4 ML
Because dataset size is constrained, Phase 4 must evaluate two distinct operational validation questions:
1. **Validation A — Unseen Tool (Group by Case / Tool ID):**
   - *Question:* If a brand new tool insert is installed under a known cutting condition (e.g., Cast Iron at DOC 0.75, Feed 0.25), can sensor features predict $V_B$ accurately?
   - *Protocol:* Leave-One-Tool-Out or GroupKFold across the 16 cases.
2. **Validation B — Unseen Cutting Condition (Group by Operating Condition):**
   - *Question:* If the machine encounters an operating condition (e.g., an unseen DOC or feed rate) that the model never saw during training, can sensor features generalize?
   - *Protocol:* Group by cutting condition (combinations of Material, DOC, Feed). This is a significantly more demanding validation test with higher industrial value.

*Reference Visualization:*
- [Figure 8: Replicate Tool Repeatability Check](file:///D:/Project/Manufacturing%20Analytics/reports/figures/phase3_feature_distributions/fig8_replicate_feature_repeatability.png)

---

## 8. Feature Leakage Audit

To guarantee scientific rigor in subsequent modeling, variables are strictly partitioned:

| Category | Variables | Permission in ML | Justification |
| :--- | :--- | :---: | :--- |
| **Valid Sensor Features** | Time- & Frequency-domain statistics from 6 channels | **PERMITTED** | Derived strictly from steady-state snapshot sensor physics during the pass. |
| **Context Features** | `material_code`, `DOC_mm`, `feed_mm_rev` | **PERMITTED (Model B Only)** | Known machine setup parameters. Useful for evaluating sensor + context fusion. |
| **Strictly Prohibited** | `run` (pass counter) | **STRICTLY PROHIBITED** | Directly proxies cumulative tool wear progression; causes catastrophic data leakage. |
| **Strictly Prohibited** | `cumulative_time_min` | **STRICTLY PROHIBITED** | Linear proxy of tool life; memorizing time bypasses sensor learning entirely. |
| **Strictly Prohibited** | `case` (tool ID) | **STRICTLY PROHIBITED (as feature)** | Used exclusively as the Grouping variable for GroupKFold cross-validation. |

---

## 9. Structured Candidate Feature Set (Organized by Purpose)

Rather than dumping an arbitrary 24-feature set into ML, candidate features are explicitly categorized by **physical purpose and modeling role**. In Phase 3.2, these candidates will undergo a **redundancy audit** (e.g., pruning `smcAC_std` due to $r=0.9999$ collinearity with `smcAC_rms`) to retain a compact, robust feature set for Phase 4:

| Sensor Channel | Feature Name | Domain | Physical Purpose | Phase 3.1 Role | Empirical Rationale & Status |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **`smcAC`** | `smcAC_rms` | Time | **Current / Load** | **Core Candidate** | Strong wear association ($r_s = 0.753$), within-tool median $r_s = 0.993$. |
| **`smcAC`** | `smcAC_std` | Time | Current / Load | Secondary (Collinear) | Identical to RMS ($r=0.9999$); candidate for pruning in Phase 3.2. |
| **`smcAC`** | `smcAC_p2p` | Time | Current / Dynamic Range | Secondary Candidate | Captures peak torque excursions ($r_s = 0.685$). |
| **`smcAC`** | `smcAC_kurtosis` | Time | Current / Shape | Secondary Candidate | Current peakedness; weak wear tracking ($r_s = -0.06$). |
| **`smcAC`** | `smcAC_crest` | Time | Current / Spikiness | Secondary Candidate | Peak to RMS ratio; normalized dynamic index. |
| **`smcAC`** | `smcAC_spindle_band_pwr` | Frequency | **Kinematics / Load** | **Core Candidate** | Spindle harmonic power ($13.8\text{ Hz}$); tracks wear ($r_s = 0.756$). |
| **`smcDC`** | `smcDC_mean` | Time | Baseline Power Draw | **Diagnostic / Conditional** | High correlation ($r_s = 0.783$), BUT **saturated in 41 runs**. Exclude from primary ML. |
| **`smcDC`** | `smcDC_std` | Time | DC Ripple Amplitude | **Diagnostic / Conditional** | Variance compressed by 10V clipping. Treat with caution. |
| **`vib_spindle`** | `vib_spindle_p2p` | Time | **Mechanical Impact** | **Core Candidate** | Captures tool edge micro-fractures ($r_s = 0.521$, within-tool $0.609$). |
| **`vib_spindle`** | `vib_spindle_kurtosis` | Time | **Mechanical Impact** | **Core Candidate** | **Wear-sensitive ($r_s = 0.591$) with near-zero DOC/Feed bias.** |
| **`vib_spindle`** | `vib_spindle_mean` | Time | Baseline Spindle Envelope | Secondary Candidate | Baseline level shifts with machine setup. |
| **`AE_table`** | `AE_table_rms` | Time | **Friction / Contact** | **Core Candidate** | Flank contact friction energy ($r_s = 0.668$, within-tool $0.957$). |
| **`AE_table`** | `AE_table_mean` | Time | Friction / Baseline | Secondary (Collinear) | Highly collinear with `AE_table_rms`. |
| **`AE_table`** | `AE_table_p2p` | Time | Acoustic Bursts | Secondary Candidate | Dynamic acoustic burst range ($r_s = 0.674$). |
| **`AE_table`** | `AE_table_kurtosis` | Time | Material Metallurgy | Diagnostic Only | Differentiates Material ($r_s = 0.750$), zero wear tracking ($r_s = 0.036$). |
| **`AE_spindle`** | `AE_spindle_p2p` | Time | **Friction / Impact** | **Core Candidate** | Tool-holder acoustic burst range ($r_s = 0.672$, lower DOC bias). |
| **`AE_spindle`** | `AE_spindle_mean` | Time | Acoustic Transmission | Secondary Candidate | Baseline tool-holder acoustic energy ($r_s = 0.573$). |
| **`AE_spindle`** | `AE_spindle_rms` | Time | Acoustic Energy | Secondary Candidate | Effective transmission energy ($r_s = 0.594$). |
| **`AE_spindle`** | `AE_spindle_kurtosis` | Time | Acoustic Peakedness | Secondary Candidate | Transient acoustic burst peakedness ($r_s = 0.257$). |
| **`vib_table`** | `vib_table_spindle_band_pwr` | Frequency | **Kinematics / Runout** | **Core Candidate** | Table vibration power at spindle frequency ($11-16.5\text{ Hz}$). |
| **`vib_table`** | `vib_table_tooth_band_pwr` | Frequency | **Kinematics / Impacts** | **Core Candidate** | Table vibration power at tooth passing frequency ($75-90\text{ Hz}$). |
| **`vib_table`** | `vib_table_mean` | Time | Table Envelope Level | Secondary Candidate | Moderate negative trend ($r_s = -0.205$, within-tool $-0.588$). |
| **`vib_table`** | `vib_table_std` | Time | Table Fluctuation | Secondary Candidate | Table envelope fluctuation amplitude ($r_s = -0.353$). |
| **`vib_table`** | `vib_table_p2p` | Time | Table Dynamics | Secondary Candidate | Strongly reflects Material ($r_s = 0.628$), negative wear trend. |

---

## 10. Answers to Core Engineering & Design Questions

### A. Should we use full-window features?
**YES.**  
The full 36-second snapshot ($N=9000$) provides high statistical stability. In-situ correlation with middle steady-state windows exceeds $r = 0.996$, confirming that full-window aggregation accurately captures dominant cutting dynamics.

### B. Should we use sub-window features?
**NO (Not for primary modeling).**  
Sub-window drift features reflect initial 5-second tool-entry stabilization rather than tool wear ($r < 0.20$ with $V_B$). Sub-windowing would triple feature dimensionality without providing independent wear information.

### C. Should we use frequency-domain features?
**YES, but strictly restricted to known kinematic harmonics.**  
We extract band powers exclusively around known physical cutting frequencies:
1. Spindle rotational band ($11.0 - 16.5\text{ Hz}$)
2. Tooth passing band ($75.0 - 90.0\text{ Hz}$)  
Blind high-dimensional FFT binning or wavelet decomposition is unjustified due to the $250\text{ Hz}$ sampling rate and prior analog RMS smoothing.

### D. Which sensors appear most promising?
1. **`smcAC` (AC Motor Current):** Strongest monotonic wear progression (within-tool median $r_s = 0.993$), reflecting cutting torque demand, but confounded by DOC and Feed.
2. **`AE_table` & `AE_spindle`:** Excellent flank friction tracking (within-tool median $r_s = 0.957$ and $0.785$). `AE_spindle_p2p` exhibits lower cutting-condition confounding.
3. **`vib_spindle` (Kurtosis & P2P):** High wear sensitivity ($r_s = 0.591$ and $0.521$) that is largely independent of cutting condition confounders.
4. **`smcDC`:** Promising correlation ($r_s = 0.783$), but degraded by hardware clipping at $+10\text{ V}$ in 41 runs. Classified as `Diagnostic / Conditional`.
5. **`vib_table`:** Primarily reflects workpiece material metallurgy rather than flank wear.

### E. Which features should proceed to Phase 3.2?
The **24 candidate features** structured in Section 9 will proceed to Phase 3.2 for **redundancy auditing and compact feature selection**.

---

## Roadmap Decision & Next Steps

> **Phase 3.1 Status:** **COMPLETED & APPROVED WITH ADJUSTMENTS.**  
> **Immediate Next Step:** **Phase 3.2 — Feature Dataset Generation + Redundancy & Within-Condition Validation.**  
> **Phase 4 (Machine Learning) Status:** **STRICTLY ON HOLD.**  

In Phase 3.2, we will:
1. Extract and serialize the audited feature dataset for all 145 valid runs.
2. Perform a collinearity and redundancy audit (e.g., removing `smcAC_std` in favor of `smcAC_rms`).
3. Formalize the compact feature set for Phase 4.
4. Set up the validation split definitions (Validation A: Unseen Tool, Validation B: Unseen Condition).

---
*Report produced by AntiGravity (Implementation Agent) under the direction of Bright and Strategic Lead ChatGPT.*
