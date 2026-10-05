# Phase 2 & Phase 2.5 Manufacturing Findings: Tool Wear Progression & Productivity-Adjusted Tool-Life Metrics
**NASA Milling Dataset (V2 Flagship Project)**

- **Project:** Manufacturing Analytics (V2)
- **Author:** AntiGravity (Implementation Agent)
- **Domain Lead / Owner:** Bright
- **Strategic Lead:** ChatGPT
- **Status:** Phase 2 & 2.5 Complete (Manufacturing Analytics & Productivity-Adjusted Layer)

---

## Executive Overview

Phase 2 focuses exclusively on **manufacturing analytics and machining process physics** across the 16 tool insert experiments in the NASA Milling dataset. In accordance with project principles, **no predictive machine learning models were constructed** in this phase. All evaluations are purely descriptive and empirically grounded in the physical cutting parameters: Workpiece Material (`Cast Iron` vs. `Stainless Steel J45`), Axial Depth of Cut (`DOC` = 0.75 mm vs. 1.50 mm), and Feed per Revolution (`feed` = 0.25 mm/rev vs. 0.50 mm/rev).

The primary analytical benchmark is designated as **$V_B = 0.30\text{ mm}$** for the purposes of this study (it is not claimed to be a universal industry-wide standard). Linear interpolation between adjacent bounding observations is employed to estimate benchmark crossing times and pass counts without fabricating data.

To avoid introducing unverified geometric assumptions (such as radial width of cut $a_e$), the official productivity metric is defined parameter-free as **Specific Material Removal Rate ($\text{MRR}' = a_p \times v_f$, $\text{mm}^2/\text{min}$)**. Building upon this, Phase 2.5 introduces the **Productivity-Adjusted Tool-Life Metric ($\text{MRR}' \times T_{0.30}$)**, quantifying the total cumulative material removed per unit cutting width before reaching the wear benchmark.

---

## A. Wear Behavior: $V_B$ Progression Over Machining Time

### 1. Trajectory Characteristics
- **Stable Progression:** Across the valid tool runs, the observed flank wear ($V_B$) progresses in a predominantly monotonic fashion over cumulative cutting time.
- **Initial Wear ($V_B$ at first observation):** 
  - Several tools exhibit non-zero $V_B$ at their earliest recorded pass (e.g., Case 2: $0.14\text{ mm}$ at $t=9.0\text{ min}$; Case 15: $0.15\text{ mm}$ at $t=3.0\text{ min}$; Case 16: $0.24\text{ mm}$ at $t=3.0\text{ min}$).
  - *Analytical Note:* Because $V_B$ measurements were collected at irregular intervals, the first recorded $V_B$ value should not necessarily be interpreted as the tool’s initial wear state.
- **Wear Acceleration (Accelerated Wear):**
  - In Cast Iron, trajectories remain relatively linear through $V_B \approx 0.50\text{ mm}$.
  - In Stainless Steel J45, particularly under aggressive cutting conditions (Cases 13, 14, 16), wear progression exhibits marked upward curvature as $V_B$ surpasses $0.40 - 0.50\text{ mm}$, culminating in severe degradation ($V_B > 1.0 - 1.53\text{ mm}$) within a few subsequent passes.
- **Interval Wear Rate ($\Delta V_B / \Delta t$):**
  - Point-to-point interval wear rates reveal that wear rate is not completely uniform throughout an insert's life. 
  - In Case 1, two consecutive measurements showed slight negative differences ($0.29 \rightarrow 0.28\text{ mm}$ at $t=26\text{ min}$, and $0.50 \rightarrow 0.44\text{ mm}$ at $t=48\text{ min}$).
  - *Analytical Note:* Negative interval changes are small relative to the overall wear progression and may reflect measurement variability or localized changes in the worn edge.

*Reference Visualization:* [Figure 1: Wear Trajectories by Condition](file:///D:/Project/Manufacturing%20Analytics/reports/figures/wear_trajectories_by_condition.png)

---

## B. Workpiece Material Effect: Cast Iron vs. Stainless Steel J45

Under the tested experimental conditions, the data show that **workpiece material is the primary differentiating factor** in observed tool life and wear progression:

1. **Condition-Matched Tool-Life Comparison (Primary Evidence):**
   - When controlling for feed and depth of cut, tools machining Cast Iron consistently achieved substantially longer cutting times before reaching $V_B = 0.30\text{ mm}$ compared to Stainless Steel J45 across all matched conditions:
     - At $\text{DOC}=0.75, f=0.25$: Cast Iron averaged **$57.6\text{ min}$** vs. Stainless Steel J45 averaged **$14.8\text{ min}$** ($\approx 3.9\times$ longer).
     - At $\text{DOC}=0.75, f=0.50$: Cast Iron averaged **$42.2\text{ min}$** vs. Stainless Steel J45 averaged **$8.3\text{ min}$** ($\approx 5.1\times$ longer).
     - At $\text{DOC}=1.50, f=0.25$: Cast Iron averaged **$22.1\text{ min}$** vs. Stainless Steel J45 reached **$6.7\text{ min}$** (Case 15; $\approx 3.3\times$ longer).
     - At $\text{DOC}=1.50, f=0.50$: Cast Iron averaged **$26.5\text{ min}$** vs. Stainless Steel J45 averaged **$5.2\text{ min}$** ($\approx 5.1\times$ longer).
2. **Interval Wear Rate Detail (Descriptive Note):**
   - Pooling all valid cutting intervals across tools, the median interval wear rate ($\Delta V_B / \Delta t$) was **$0.0080\text{ mm/min}$** in Cast Iron (80 intervals) and **$0.0408\text{ mm/min}$** in Stainless Steel J45 (47 intervals). While this pooled ratio reflects roughly 5.1×, individual tool observation frequencies and test durations vary; hence condition-matched tool life (Table E) serves as our primary comparative baseline.
3. **Physical Context (Domain Explanation):**
   - Austenitic stainless steels typically have lower thermal conductivity and higher work-hardening propensity than pearlitic/ferritic gray cast irons. Frictional heating at the tool–chip interface remains localized at the cutting tip, which is consistent with the rapid wear escalation observed on the KC710 coated carbide inserts.

*Reference Visualization:* [Figure 2: Material Wear Comparison](file:///D:/Project/Manufacturing%20Analytics/reports/figures/material_wear_comparison.png)

---

## C. Feed Rate Effect: 0.25 mm/rev vs. 0.50 mm/rev

*Note on Feed Units:* The feed parameter in the NASA dataset is expressed as feed per revolution ($f_{\text{rev}}$ in $\text{mm/rev}$), corresponding to table feed velocities of $v_f = 206.5\text{ mm/min}$ (at $f=0.25\text{ mm/rev}$) and $v_f = 413.0\text{ mm/min}$ (at $f=0.50\text{ mm/rev}$) at the fixed spindle speed of $826\text{ RPM}$.

1. **Wear Trajectory Slope:**
   - In both materials, increasing feed from $0.25$ to $0.50\text{ mm/rev}$ is associated with steeper wear trajectories over machining time.
   - In Cast Iron at $\text{DOC} = 0.75\text{ mm}$, estimated time to $V_B = 0.30\text{ mm}$ decreased from an average of **57.6 min** ($f=0.25$) to **42.2 min** ($f=0.50$) — a 26.7% reduction in tool cutting time.
   - In Stainless Steel J45 at $\text{DOC} = 0.75\text{ mm}$, estimated time to $V_B = 0.30\text{ mm}$ decreased from an average of **14.8 min** ($f=0.25$) to **8.3 min** ($f=0.50$) — a 43.9% reduction in tool cutting time.
2. **Productivity Coupling (Pass-Count Decoupling):**
   - While increasing feed accelerates wear rate per minute of cutting, it simultaneously doubles the linear feed speed ($v_f = 413\text{ vs } 206.5\text{ mm/min}$), halving the required machining time per 483 mm pass ($1.17\text{ vs } 2.34\text{ min}$).
   - Consequently, the number of completed passes before reaching $0.30\text{ mm}$ does not decrease by half: for Cast Iron at $\text{DOC} = 0.75\text{ mm}$, tools completed an average of **13.0 passes** at $0.25\text{ mm/rev}$ and **9.3 passes** at $0.50\text{ mm/rev}$.

*Reference Visualization:* [Figure 3: Feed Wear Comparison](file:///D:/Project/Manufacturing%20Analytics/reports/figures/feed_wear_comparison.png)

---

## D. Depth of Cut (DOC) Effect: 0.75 mm vs. 1.50 mm

1. **Observed Wear Progression:**
   - Under the tested conditions, increasing DOC from $0.75\text{ mm}$ to $1.50\text{ mm}$ is consistently associated with a substantial reduction in cumulative cutting time before reaching $V_B = 0.30\text{ mm}$.
   - In Cast Iron at $f=0.25\text{ mm/rev}$:
     - $\text{DOC} = 0.75\text{ mm}$ (Cases 3, 11): Mean time to $0.30\text{ mm}$ = **57.6 min** (13.0 passes).
     - $\text{DOC} = 1.50\text{ mm}$ (Cases 4, 10): Mean time to $0.30\text{ mm}$ = **22.1 min** (4.7 passes).
     - Tool life in minutes decreased by **61.6%**.
2. **Interaction with Material:**
   - In Stainless Steel J45 at $f=0.25\text{ mm/rev}$, increasing DOC from 0.75 to 1.50 mm reduced tool life from **14.8 min** (5.7 passes) to **6.7 min** (3.2 passes in Case 15; Case 6 aborted).

*Reference Visualization:* [Figure 4: DOC Wear Comparison](file:///D:/Project/Manufacturing%20Analytics/reports/figures/doc_wear_comparison.png)

---

## E. Tool-Life Benchmark Table ($V_B = 0.30\text{ mm}$)

Tool life was evaluated across three wear thresholds using linear interpolation between bounding observations:

| Condition | Material | DOC (mm) | Feed (mm/rev) | Tool Rep 1 | Tool Rep 2 | Time to 0.30 mm (min) [Rep 1 / Rep 2] | Mean Time to 0.30 mm (min) | Time to 0.50 mm (min) [Rep 1 / Rep 2] | Time to 0.80 mm (min) [Rep 1 / Rep 2] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Cast Iron | 0.75 | 0.25 | Case 3 | Case 11 | 59.4 / 55.8 | **57.6** $\pm$ 2.5 | 78.3 / 81.8 | Censored / Censored |
| **C2** | Cast Iron | 0.75 | 0.50 | Case 2 | Case 12 | 43.8 / 40.5 | **42.2** $\pm$ 2.3 | 67.7 / 61.7 | Censored / Censored |
| **C3** | Cast Iron | 1.50 | 0.25 | Case 4 | Case 10 | 20.5 / 23.7 | **22.1** $\pm$ 2.3 | Censored* / 42.0 | Censored / Censored |
| **C4** | Cast Iron | 1.50 | 0.50 | Case 1 | Case 9 | 29.3 / 23.6 | **26.5** $\pm$ 4.0 | 44.0 / 35.1 | Censored / 45.6 |
| **C5** | Stainless Steel J45 | 0.75 | 0.25 | Case 7 | Case 13 | 14.2 / 15.3 | **14.8** $\pm$ 0.8 | Censored* / 22.4 | Censored / 31.4 |
| **C6** | Stainless Steel J45 | 0.75 | 0.50 | Case 8 | Case 14 | 6.0 / 10.6 | **8.3** $\pm$ 3.3 | 10.0 / 15.6 | Censored / 20.9 |
| **C7** | Stainless Steel J45 | 1.50 | 0.25 | Case 6 | Case 15 | Aborted / 6.7 | **6.7** (Rep 2) | Aborted / 13.8 | Aborted / Censored |
| **C8** | Stainless Steel J45 | 1.50 | 0.50 | Case 5 | Case 16 | 6.2 / 4.1 | **5.2** $\pm$ 1.5 | 11.0 / 7.4 | Censored / Censored |

*\*Note on Right-Censoring:* Case 4 was terminated at $V_B = 0.49\text{ mm}$ and Case 7 was terminated at $V_B = 0.46\text{ mm}$.  
*\*Note on 0.80 mm Threshold:* Only 3 of 16 tools (Cases 9, 13, 14) actually reached $V_B \ge 0.80\text{ mm}$. The remaining 13 tools were stopped before reaching this wear level. Consequently, **0.80 mm cannot serve as a reliable comparative benchmark** across the dataset.

*Full Dataset Table:* [CSV: `data/phase2_tool_life_metrics.csv`](file:///D:/Project/Manufacturing%20Analytics/data/phase2_tool_life_metrics.csv)  
*Reference Visualization:* [Figure 5: Tool Life Comparison](file:///D:/Project/Manufacturing%20Analytics/reports/figures/tool_life_comparison.png)

---

## F. Productivity vs. Tool-Life Trade-Off & Phase 2.5 Metrics

### 1. Official Productivity Metric Definition
To maintain absolute data integrity without fabricating radial cutting width ($a_e$), the official productivity metric is defined strictly through verified parameters:
$$\text{MRR}' = a_p \times v_f \quad (\text{mm}^2/\text{min})$$
where:
- Spindle speed $n = 826\text{ RPM}$
- Linear feed velocity $v_f = f_{\text{rev}} \times n$ ($206.5\text{ mm/min}$ at $f=0.25$; $413.0\text{ mm/min}$ at $f=0.50$)
- Axial depth of cut $a_p \in \{0.75, 1.50\}\text{ mm}$

Across the tested conditions:
- Condition 1 & 5 ($a_p=0.75, f=0.25$): $\text{MRR}' = 154.875\text{ mm}^2/\text{min}$ ($1\times$)
- Condition 2 & 6 ($a_p=0.75, f=0.50$): $\text{MRR}' = 309.750\text{ mm}^2/\text{min}$ ($2\times$)
- Condition 3 & 7 ($a_p=1.50, f=0.25$): $\text{MRR}' = 309.750\text{ mm}^2/\text{min}$ ($2\times$)
- Condition 4 & 8 ($a_p=1.50, f=0.50$): $\text{MRR}' = 619.500\text{ mm}^2/\text{min}$ ($4\times$)

### 2. Phase 2.5: Productivity-Adjusted Tool-Life Metric
Rather than evaluating tool life in minutes alone, Phase 2.5 asks the fundamental manufacturing question:
> **"How much cumulative material does each cutting strategy remove before reaching the wear benchmark?"**

We define the **Productivity-Adjusted Tool Life**:
$$\text{Cumulative Material Removed Proxy} = \text{MRR}' \times T_{0.30} \quad (\text{mm}^2 \text{ per unit cutting width})$$

| Condition | Material | DOC (mm) | Feed (mm/rev) | $\text{MRR}'$ ($\text{mm}^2/\text{min}$) | Mean $T_{0.30}$ (min) | Mean Material Removed Proxy before $0.30\text{ mm}$ ($\text{mm}^2$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **C1** | Cast Iron | 0.75 | 0.25 | 154.9 | 57.6 | **8,921 $\text{mm}^2$** |
| **C2** | Cast Iron | 0.75 | 0.50 | 309.8 | 42.2 | **13,056 $\text{mm}^2$** |
| **C3** | Cast Iron | 1.50 | 0.25 | 309.8 | 22.1 | **6,843 $\text{mm}^2$** |
| **C4** | Cast Iron | 1.50 | 0.50 | 619.5 | 26.5 | **16,407 $\text{mm}^2$** |
| **C5** | Stainless Steel J45 | 0.75 | 0.25 | 154.9 | 14.8 | **2,287 $\text{mm}^2$** |
| **C6** | Stainless Steel J45 | 0.75 | 0.50 | 309.8 | 8.3 | **2,577 $\text{mm}^2$** |
| **C7** | Stainless Steel J45 | 1.50 | 0.25 | 309.8 | 6.7 | **2,065 $\text{mm}^2$** |
| **C8** | Stainless Steel J45 | 1.50 | 0.50 | 619.5 | 5.2 | **3,198 $\text{mm}^2$** |

### 3. The Equal-Productivity Strategic Comparison ($\text{MRR}' = 309.8\text{ mm}^2/\text{min}$)
Conditions 2 and 3 operate at **identical specific material-removal-rate proxy** ($\text{MRR}' = a_p \times v_f = 309.75\text{ mm}^2/\text{min}$):
- **Condition 2 (High Feed / Low DOC):** $a_p = 0.75\text{ mm}, f = 0.50\text{ mm/rev} \implies \text{Cumulative Material-Removal Proxy} = \mathbf{13,056\text{ mm}^2}$ per unit width
- **Condition 3 (Low Feed / High DOC):** $a_p = 1.50\text{ mm}, f = 0.25\text{ mm/rev} \implies \text{Cumulative Material-Removal Proxy} = \mathbf{6,843\text{ mm}^2}$ per unit width

**Strategic Finding:**  
At the same specific material-removal-rate proxy, **the High Feed / Low DOC condition achieved approximately $1.91\times$ higher cumulative material-removal proxy before reaching $V_B = 0.30\text{ mm}$** compared to the Low Feed / High DOC condition.

*(Analytical Clarification: The metric $\text{MRR}' \times T_{0.30}$ (expressed in $\text{mm}^2$ per unit cutting width) is an empirical productivity-adjusted tool-life proxy reflecting estimated relative cutting capability under the tested kinematic parameters, rather than a physically weighed or optically measured volumetric workpiece loss.)*

*Reference Visualization:* [Figure 6: Productivity vs Tool Life](file:///D:/Project/Manufacturing%20Analytics/reports/figures/productivity_vs_tool_life.png)

---

## G. Replicate Consistency: Repeatability Between Tested Tools

Each cutting condition was tested with two distinct inserts (Replicate 1: Cases 1–8; Replicate 2: Cases 9–16):

1. **Strong Ordinal Agreement:**
   - In all conditions with valid pairs, Replicate 1 and Replicate 2 exhibited consistent relative rankings: conditions that degraded rapidly in Replicate 1 did so in Replicate 2.
2. **Quantitative Discrepancies:**
   - **Condition 1 (Cast Iron, d0.75 f0.25):** Case 3 reached $0.30\text{ mm}$ at $59.4\text{ min}$; Case 11 at $55.8\text{ min}$ (difference: 6.1%).
   - **Condition 2 (Cast Iron, d0.75 f0.50):** Case 2 reached $0.30\text{ mm}$ at $43.8\text{ min}$; Case 12 at $40.5\text{ min}$ (difference: 7.5%).
   - **Condition 4 (Cast Iron, d1.50 f0.50):** Case 1 reached $0.30\text{ mm}$ at $29.3\text{ min}$; Case 9 at $23.6\text{ min}$ (difference: 19.5%).
   - **Condition 6 (Stainless Steel J45, d0.75 f0.50):** Case 8 reached $0.30\text{ mm}$ at $6.0\text{ min}$; Case 14 at $10.6\text{ min}$ (difference: 43.4%).
3. **Analytical Assessment:**
   - The broad manufacturing degradation patterns are reasonably reproducible.
   - However, with only $N=2$ replicates per condition, random microstructural tool variations, coating integrity differences, and offline measurement noise introduce non-negligible variance. High-confidence statistical claims of equivalence are avoided.

---

## H. Data Quality Flags & Limitations

To ensure research integrity, the following limitations are documented:

1. **Small Sample Size ($N=16$ Tools):**
   - The dataset contains exactly 16 tool insert runs across 8 conditions (2 replicates each). This is a classical experimental design, not a large-scale industrial observation study.
2. **Case 6 Abortion:**
   - Case 6 (Stainless Steel J45, $\text{DOC}=1.50\text{ mm}, f=0.25\text{ mm/rev}$) was aborted after a single pass at $t=0.0\text{ min}, V_B = 0.00\text{ mm}$. It provides zero wear progression information and is excluded from aggregated tool-life averages.
3. **Corrupted Sensor Signals in Valid Tool Runs:**
   - Case 2 Run 1 has a measured $V_B = 0.08\text{ mm}$ but a corrupted DAQ signal ($>10^{19}$).
   - Case 12 Run 1 has both a corrupted sensor recording and missing $V_B$.
4. **Offline and Irregular Wear Measurements:**
   - Tool wear was not measured continuously in-situ. The insert was unmounted and measured optically under a microscope at irregular intervals. Incomplete passes exist in 20 runs.
5. **Fixed Cutting Speed ($v_c = 200\text{ m/min}$):**
   - Spindle speed was held fixed at 826 RPM throughout all 167 runs. The classic Taylor tool-life speed exponent ($v_c T^n = C$) cannot be directly fitted from this dataset.
6. **Severe Right-Censoring at $V_B \ge 0.80\text{ mm}$:**
   - 13 of 16 tools never reached $0.80\text{ mm}$ because the experimenters stopped cutting once standard tool retirement wear was confirmed.
7. **No Causal Inferences:**
   - All relationships reported herein represent observed empirical associations under the Matsuura MC-510V setup and do not constitute universal machining laws.

---

## Revised Project Roadmap & Next Steps

Rather than building speculative business cost models based on assumed labor, machine, and tooling prices (which do not exist in the NASA dataset), the project follows a disciplined data science progression:

```
[Phase 1: Signal & Data Foundation] ✅
       │
[Phase 2: Manufacturing Analytics Layer] ✅
       │
[Phase 2.5: Productivity-Adjusted Tool Life (MRR' × T_0.30)] ✅
       │
       ▼
[Phase 3: Sensor Signal Feature Engineering] 🔜
  • 6 Sensor Channels: smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle
  • Time Domain: RMS, Peak-to-Peak, Crest Factor, Kurtosis, Skewness
  • Frequency Domain: Spectral Centroid, Band Power, Dominant Peak
  • Time-Frequency: Wavelet / Spectrogram energy bands
       │
       ▼
[Phase 4: Machine Learning Tool Wear Prediction]
  • GroupKFold by Tool/Case (strict zero-data-leakage validation)
  • Baseline (Ridge / SVR) → Ensemble (Random Forest / XGBoost)
  • Model Interpretation: Feature Importance & SHAP Analysis
```

---
*Report produced by AntiGravity (Implementation Agent) under the direction of Bright and Strategic Lead ChatGPT.*
