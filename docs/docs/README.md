# RunwaySafe Documentation Suite: Master Index & Executive Overview

Welcome to the official documentation suite for **RunwaySafe: Evidence-Based Runway Risk Assessment & Operational Decision Support**.

This documentation suite has been engineered specifically to provide complete end-to-end clarity for both non-technical stakeholders (airport executives, airfield duty managers) and technical specialists (data scientists, software architects, aviation safety auditors).

---

## Documentation Structure

```
RunwaySafe/docs/
├── README.md                                  # (This Master Index)
├── 01_problem_statement_and_gap_analysis.md   # Problem domain, current industry solutions & the innovation gap
├── 02_dashboard_walkthrough_and_user_guide.md # Non-technical "unboxing-style" UI tour & terminology guide
├── 03_mathematical_calculations_and_formulas.md# Detailed mathematical formulas, weights & scoring models
├── 04_technical_architecture_and_ml_pipeline.md# Full technical deep-dive: preprocessing, training & code analysis
└── 05_updations_audit_and_improvement_roadmap.md# Comprehensive system audit, calculation flaws & upgrade roadmap
```

---

## Document Summary & Navigation Guide

### 📘 [01. Problem Statement & Industry Gap Analysis](file:///home/yaazh/Projects/RunwaySafe/docs/01_problem_statement_and_gap_analysis.md)
* **Audience:** Airport Executives, Aviation Safety Officers, Client Stakeholders.
* **Key Topics:**
  * Real-world runway hazards (hydroplaning, excursions, rubber friction loss, traffic congestion).
  * Critique of existing industry methods (ICAO GRF, AWOS sensors, flight databases, SMS logbooks).
  * How RunwaySafe bridges the gap by synthesizing 4 isolated data streams into unified predictive intelligence.

---

### 📦 [02. Dashboard Walkthrough & User Guide ("The Unboxing Tour")](file:///home/yaazh/Projects/RunwaySafe/docs/02_dashboard_walkthrough_and_user_guide.md)
* **Audience:** Airfield Duty Managers, Tower Controllers, Field Inspectors.
* **Key Topics:**
  * Clean, unboxing-style walkthrough of all 9 dashboard tabs:
    1. *Home / Overview:* Airport fleet risk pulse check.
    2. *Runway Risk Assessment:* Diagnostic deep-dive station.
    3. *What-if Risk Calculator:* User-entered scenario risk calculation.
    4. *Current Conditions:* Physical inspection sensor readings.
    5. *Safety Alerts:* Real-time threshold warnings.
    6. *Recommendations:* Actionable maintenance advisories.
    7. *Risk History:* Longitudinal incident trend analysis.
    8. *Analytics:* Multi-domain business intelligence.
    9. *Reports:* Official CSV audit compliance exports.
  * Every metric, badge (🟢/🟠/🔴), dropdown, and term translated into plain aviation language.

---

### 📐 [03. Mathematical Calculations & Formulas](file:///home/yaazh/Projects/RunwaySafe/docs/03_mathematical_calculations_and_formulas.md)
* **Audience:** Safety Analysts, Operations Researchers, Technical Auditors.
* **Key Topics:**
  * Complete mathematical formulations of the AIRS-RAN index.
  * Equations for Weather Risk ($R_{\text{weather}}$), Runway Condition Risk ($R_{\text{runway}}$), Operational Risk ($R_{\text{operational}}$), and Historical Risk ($R_{\text{historical}}$).
  * Dynamic weighting shift algorithms for adverse weather and traffic surges.
  * Machine Learning probability equations and confidence calculations.

---

### ⚙️ [04. Technical Architecture & Machine Learning Pipeline](file:///home/yaazh/Projects/RunwaySafe/docs/04_technical_architecture_and_ml_pipeline.md)
* **Audience:** Data Scientists, ML Engineers, Backend Developers.
* **Key Topics:**
  * Data preprocessing across 4 multi-tabular datasets (`preprocessing.py`).
  * Prevention of temporal target leakage during cumulative incident feature extraction.
  * Scikit-Learn `ColumnTransformer` (Median Imputation with Indicators, StandardScaler, OneHotEncoding).
  * Machine learning tournament: Decision Tree vs. Random Forest (300 estimators, balanced class weights).
  * Empirical benchmark results: 82.5% accuracy, 0.812 ROC-AUC, and feature importance rankings with corresponding code references.

---

### 🚀 [05. System Audit, Calibration Gaps & Roadmap of Upgrades](file:///home/yaazh/Projects/RunwaySafe/docs/05_updations_audit_and_improvement_roadmap.md)
* **Audience:** Project Managers, Lead Developers, Product Owners.
* **Key Topics:**
  * In-depth audit of current calculation issues (weight decrement anomalies, metric saturation).
  * Addressing client confusion between ML categorical predictions and AIRS-RAN scores.
  * Roadmap for Phase 1 (Mathematical Re-calibration), Phase 2 (XGBoost, SMOTE & SHAP Explainability), and Phase 3 (Interactive "What-If" Simulation Playground and Spatial Airfield Maps).
  * Concrete priority implementation matrix (P0 to P2).
