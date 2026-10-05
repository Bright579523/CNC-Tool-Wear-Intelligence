# V2 Technical Design Specification: NASA Milling Wear Analytics

**Project Title:** Data-Driven CNC Machining & Tool Wear Analytics  
**Strategic Lead:** ChatGPT  
**Implementation Agent:** AntiGravity  
**Owner:** Bright  
**Status:** Approved & Locked Blueprint  

---

## 1. NASA Variables Specification

### Core Input Variables
* `material`: Cast Iron (1) vs Stainless Steel J45 (2)
* `feed`: $0.25$ or $0.50\text{ mm/rev}$ (Feed per revolution)
* `DOC` / $a_p$: $0.75$ or $1.50\text{ mm}$ (Axial depth of cut)
* `case`: Independent tool insert ID (1 to 16) — **Group Key only, never ML feature**
* `smcAC`, `smcDC`: Motor currents (AC/DC)
* `vib_table`, `vib_spindle`: Vibration signals
* `AE_table`, `AE_spindle`: Acoustic emission signals

### Strictly Excluded from ML Input (Anti-Leakage Rule)
* `run`: Pass sequence number (proxy for tool age)
* `cumulative_time`: Elapsed cutting minutes (proxy for wear progression)
* `VB`: Ground truth wear target

---

## 2. Multi-Level Targets & Metrics

1. **Target A (Tool Wear):** Continuous flank wear $V_B\text{ (mm)}$
2. **Target B (Wear Rate):** $\text{Wear Rate} = \frac{\Delta V_B}{\Delta t_{cut}}\text{ (mm/min)}$
3. **Tool Life Metric:** Cumulative cutting time (and passes) to reach analysis threshold $V_B = 0.30\text{ mm}$ (sensitivity at $0.50$ and $0.80\text{ mm}$)
4. **Productivity Metric:**
   $$\text{MRR} = a_p \times a_e \times f_{rev} \times n \quad (\text{mm}^3/\text{min})$$
5. **Machining Cost Proxy:**
   $$C_{total} = C_m T_{machining} + C_t N_{tools} + C_{tc} N_{changes}$$
   Evaluated across 3 sensitivity scenarios (High tool cost, High machine rate, Long tool change).

---

## 3. Machine Learning Architecture

* **Model A (Sensor-Only):** Signals $\rightarrow$ Estimated $V_B$
* **Model B (Sensor + Context):** Signals + Material + Feed + DOC $\rightarrow$ Estimated $V_B$
* **Validation Strategy:** Strict **GroupKFold (Leave-One-Tool-Out)** grouped by `case`
* **Models:** Mean Baseline, Ridge Regression, Random Forest, XGBoost (No deep learning)
* **Metrics:** MAE, RMSE, $R^2$, and Error relative to $0.30\text{ mm}$ threshold.
