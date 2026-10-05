# CNC Tool Wear Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20Milling%20(PCoE)-green.svg)](https://www.nasa.gov/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success.svg)](https://bright579523.github.io/CNC-Tool-Wear-Intelligence/)

A data science project for estimating flank wear in CNC milling using machining conditions and sensor measurements.

The project uses the NASA Milling Dataset to study how cutting conditions and sensor signals relate to tool wear, then builds a regression model to estimate flank wear width $VB$.

A browser dashboard is included to make the model easier to understand and use.

---

## What this project does

The analysis follows the workflow of a real manufacturing data problem:
1. Inspect the experimental data and tool wear progression
2. Compare wear across machining conditions
3. Analyse sensor signals and extract useful features
4. Compare regression models
5. Test generalisation to unseen tool cases and machining conditions
6. Analyse model behaviour and prediction errors
7. Build a simple interface for wear estimation

---

## Dataset

* **Dataset:** NASA Milling Dataset (UC Berkeley BEST Lab / NASA Ames PCoE)
* **Machine:** Matsuura MC-510V milling machine
* **Tooling:** KC710 coated carbide inserts
* **Workpiece materials:** Cast Iron and Stainless Steel J45
* **Cutting speed:** 200 m/min
* **Feed rates:** 0.25 and 0.50 mm/rev
* **Depth of cut:** 0.75 and 1.50 mm
* **Sensor channels:** Six sensory channels
* **Scope:** 16 tool cases, 145 valid labelled observations

The target variable is flank wear width:
* `VB_mm`

The analysis uses $VB = 0.30\text{ mm}$ as an analytical tool life threshold motivated by the uniform flank wear criterion described in ISO 8688-2 for milling.

---

## Sensor features

Five sensor features were selected for the final model:
* `smcAC_rms`
* `vib_spindle_kurtosis`
* `vib_spindle_p2p`
* `AE_table_rms`
* `AE_spindle_p2p`

These are combined with three machining context variables:
* `material_code`
* `DOC_mm`
* `feed_mm_rev`

---

## Model and validation

* **Primary model:** Ridge Regression with context fusion.
* **Formulation:** Standardised inputs, fitted with $\alpha = 1.0$.
* **Data splitting:** Random train-test splitting was avoided because multiple observations come from the same physical tools.

Two validation strategies were used:
* **Leave-One-Case-Out (LOGO):** Tests performance on an unseen tool case.
* **Leave-One-Condition-Out (LOCO):** Tests performance on an unseen combination of the tested machining factors.

---

## Results

| Model Setup | Validation Strategy | MAE |
| :--- | :--- | :---: |
| Sensor-only Ridge | LOGO | 0.1361 mm |
| **Ridge with machining context** | **LOGO** | **0.1044 mm** |
| **Ridge with machining context** | **LOCO** | **0.1221 mm** |

* The context-based model reduced LOGO MAE by about **47.7%** compared with the mean baseline (and by **23.3%** compared with sensor-only Ridge).
* SVR achieved the lowest average LOCO MAE at 0.1149 mm, but **Ridge was selected as the primary model** because it provides a simpler and more directly interpretable formulation while maintaining strong performance.
* **GBDT** is included as a supporting nonlinear model.

---

## Dashboard

* **Live Demo:** [Open Interactive Dashboard on GitHub Pages](https://bright579523.github.io/CNC-Tool-Wear-Intelligence/)
* **Local file:** [`index.html`](index.html)

The dashboard allows users to:
* Enter machining conditions
* Enter sensor measurements
* Estimate flank wear width $VB$
* Compare the result with the analytical 0.30 mm threshold
* Test an alternative machining condition using the same sensor measurements

Key implementation details:
* The interface is available in **English** and **German**.
* The dashboard uses the saved Ridge model artifact in [`data/ridge_model.json`](data/ridge_model.json).
* The browser calculation was checked against the Python implementation and the model artifact, confirming end-to-end parity to machine precision ($\le 2.22 \times 10^{-16}\text{ mm}$).

---

## Key findings

* Machining context improves wear estimation compared with using sensor features alone.
* `smcAC_rms` receives the largest linear weight in the final Ridge model.
* Prediction error becomes larger at higher observed wear.
* The highest wear observation is difficult for both models to estimate accurately.
* Model performance should be interpreted within the range and conditions represented in this dataset.

---

## Limitations

* This is a **portfolio study based on a small experimental dataset**.
* The model is **not intended to be used as a production tool replacement system, an emergency failure detector, or a universal tool life model**.
* The results are limited to the experimental conditions represented in the NASA dataset.

---

## Reproduce

1. Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

2. Verify model and dashboard parity:
```bash
python scripts/verify_dashboard_parity.py
```

3. Then open `index.html` in a modern browser.

---

## Project status

The analysis, model selection, validation, dashboard design, and reproducibility checks are complete.
