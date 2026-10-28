# RunwaySafe: System Audit, Calibration Gaps & Roadmap of Required Upgrades

## 1. Executive Summary

While the RunwaySafe project establishes a solid foundation by joining multiple tabular aviation datasets and training a Random Forest baseline, **an in-depth engineering and UI/UX audit reveals critical operational gaps, calculation misalignments, and dashboard limitations**.

Currently, the dashboard does not clearly communicate its dual prediction mechanism to non-technical users, certain mathematical components have scaling inconsistencies, and the UI lacks real-time interactive simulation capabilities. This document details all identified deficiencies, provides their root causes, and presents a concrete implementation roadmap to elevate RunwaySafe into an enterprise-grade aviation decision-support suite.

---

## 2. In-Depth Audit of Current Deficiencies

### 2.1 The "Dual Score" Cognitive Confusion (ML Label vs. AIRS-RAN Score)
* **Observed Issue:** On Tab 2 ("Runway Risk Assessment"), users are presented with two distinct primary outputs side-by-side:
  1. `ML Prediction` (e.g., `"Low"`, Confidence: `74%`)
  2. `Overall Runway Risk` (e.g., `"68/100 🔴 High"`)
* **Impact on Client:** Clients and airfield operators are immediately confused when the Machine Learning model says `Low Risk` while the AIRS-RAN indicator flashes `High Risk`. 
* **Root Cause:** 
  * The Random Forest classifier was trained on historical incident outcomes (which are sparse and heavily weighted toward Low-risk days).
  * The AIRS-RAN score is an empirical rule-based formula that penalizes instantaneous threshold violations (like heavy rain or poor friction).
  * The dashboard does not explain *why* or *how* these two complement each other (e.g., "AI Baseline vs. Dynamic Stress Index").

### 2.2 Mathematical Calibration Flaws in Risk Calculations (`prediction.py`)

#### A. Component Score Saturation & Clipping
* In `_risk_components()`:
  ```python
  # Runway risk calculation in prediction.py
  max(0.0, 0.7 - float(row["friction_coefficient"])) * 100.0
  ```
  If friction is `0.30` (severely slippery runway), this single term adds $(0.7 - 0.3) \times 100 = 40.0$ points. If water accumulation is `8.0 mm`, it adds $8 \times 3 = 24.0$ points. If debris is `2.0`, it adds $20.0$ points. Surface condition "Poor" adds $18.0$ points.
  $$\text{Total} = 40 + 24 + 20 + 18 = 102.0 \implies \text{Clamped to } 100.0$$
* **The Problem:** The formula frequently saturates and maxes out at 100 on moderately rainy days, losing granular distinction between "Moderately Wet" and "Critically Flooded".

#### B. The Dynamic Weighting Normalization Bug
* In `_airs_ran()` (lines 165-181 of `prediction.py`):
  ```python
  weights = {"Weather Risk": 0.25, "Runway Condition Risk": 0.35, "Flight Operations Risk": 0.25, "Historical Safety Risk": 0.15}
  if row["weather_severity"] in {"Adverse", "Severe"}:
      weights["Weather Risk"] += 0.10
      weights["Runway Condition Risk"] -= 0.05
      weights["Flight Operations Risk"] -= 0.05
  if float(row["traffic_density_mean"]) >= 70:
      weights["Flight Operations Risk"] += 0.10
      weights["Runway Condition Risk"] -= 0.05
      weights["Historical Safety Risk"] -= 0.05
  ```
* **The Flaw:** If **both** Severe Weather and High Traffic Density happen concurrently, `Runway Condition Risk` is penalized twice ($-0.05 - 0.05 = -0.10$), dropping from $0.35$ down to $0.25$! In aviation reality, when it is raining heavily and the airport is congested, runway pavement safety becomes *more* critical, not less. The weighting matrix needs a mathematically rigorous normalized vector function $\mathbf{w} / \sum w_i$ rather than hardcoded decrements.

#### C. Top Contributing Factors Misalignment
* In `_factor_explanations()`, the code evaluates factors by arbitrary multiplication weights (e.g., `visibility * 4`, `friction * 100`) rather than using true Shapley values (SHAP) or normalized z-score contributions from the ML model.

### 2.3 Machine Learning Limitations (`ml_training.py`)

1. **Weak Performance on Minority Classes:**
   * In `results/evaluation_results.json`:
     * Precision for **Medium Risk**: **27.0%** (F1-score: **0.274**)
     * Recall for **High Risk**: **58.8%** (41.2% of high-risk cases missed)
   * The current model misses over 4 out of 10 high-risk events. In commercial aviation safety, a false negative (failing to detect a high risk) is intolerable.
2. **Lack of Advanced Ensembles (e.g., XGBoost / LightGBM):**
   * Only basic `DecisionTreeClassifier` and `RandomForestClassifier` are tested. Gradient boosting models with focal loss or SMOTE oversampling would significantly improve high-risk detection.
3. **Absence of K-Fold Cross Validation:**
   * Currently, training relies on a single fixed 80/20 train/test split. Cross-validation is mandatory to prevent overfitting to specific airport distributions.

### 2.4 Dashboard UI/UX & Information Architecture Flaws (`app.py`)

1. **Static Historical Playback Only (addressed):**
   * The dashboard now includes a What-if Risk Calculator. Operators can edit weather, runway, traffic, delay, flight-count, and incident-history inputs, then calculate a scenario without modifying stored data.
2. **Missing Runway Spatial Map / Airfield Diagram:**
   * Everything is rendered as flat tables and bar charts. There is no visual representation of the airport runway layout showing which runway threshold is red or green.
3. **No Model Explainability (SHAP / TreeSHAP):**
   * The dashboard lacks local explainability waterfall plots showing pilots and inspectors why a specific prediction was made.

---

## 3. Comprehensive Upgrade Roadmap

Below is the step-by-step master plan for updating and modernizing RunwaySafe:

```
+-----------------------------------------------------------------------------------------------------+
|                                 RUNWAYSAFE UPGRADE ROADMAP                                         |
+------------------------------------+--------------------------------+-------------------------------+
|       PHASE 1: Core Precision      |    PHASE 2: Advanced ML/AI     |     PHASE 3: UX & Simulation   |
|   (Mathematical Re-calibration)   |  (Gradient Boosting & SHAP)    |   (What-If Mode & Mapping)    |
+------------------------------------+--------------------------------+-------------------------------+
| • Normalize dynamic weights vector | • Implement XGBoost / LightGBM | • Add "What-If" Simulation Tab|
| • Non-linear sigmoidal risk curves | • SMOTE minority oversampling  | • Visual Airfield Runway Map  |
| • Unified AI+Rule Composite Index  | • Calibrated probabilities     | • PDF/Executive Alert Exports |
| • Eliminate hardcoded decrements   | • SHAP waterfall explanations  | • Auto-refresh live data feed |
+------------------------------------+--------------------------------+-------------------------------+
```

### Phase 1: Mathematical & Algorithmic Re-Calibration

#### 1. Implement Normalized Softmax Dynamic Weighting
Instead of subtracting fixed percentages, use a normalized Dirichlet or softmax formulation:
$$w_i = \frac{w_i^{\text{base}} \times e^{\gamma_i \cdot \text{threat}_i}}{\sum_{j=1}^{4} w_j^{\text{base}} \times e^{\gamma_j \cdot \text{threat}_j}}$$
This guarantees that weights always sum strictly to $1.00$ while escalating the priority of the most stressed subsystem without artificially depressing physical runway pavement importance.

#### 2. Replace Linear Clamping with Sigmoidal Continuous Curves
Replace linear multiplications ($\text{water} \times 3$) with continuous logistics functions:
$$R_{\text{water}} = \frac{100}{1 + e^{-k(\text{water} - x_0)}}$$
This mirrors real hydroplaning dynamics (risk stays low until 3 mm, spikes rapidly between 3 mm and 9 mm, and smoothly asymptotes at 100).

#### 3. Unify the "Dual Output" into a Single Cohesive Safety Indicator
Create a single primary metric: **The Integrated Runway Threat Level (IRTL)**:
$$\text{IRTL} = \alpha \cdot P(\text{High Risk} \mid \mathbf{x}) \times 100 + (1 - \alpha) \cdot S_{\text{AIRS-RAN}}$$
With clear contextual labels:
* *AI Historical Baseline Hazard:* Evaluated from 1,800+ past flights & incidents.
* *Instantaneous Physical Stress:* Evaluated from live sensors and inspection gages.

---

### Phase 2: Machine Learning Upgrades

#### 1. Adopt Modern Gradient Boosting (XGBoost / LightGBM / CatBoost)
* Transition from `RandomForestClassifier` to `XGBClassifier` with custom cost-sensitive loss.
* Set false-negative penalty factor $C_{\text{FN}} = 5 \times C_{\text{FP}}$ so the model aggressively avoids missing high-risk runway conditions.

#### 2. Apply SMOTE-NC (Synthetic Minority Over-sampling)
* Resample the training dataset to balance the minority classes (`Medium: 178`, `High: 88`) against `Low: 1534` to boost high-risk recall above **85%**.

#### 3. Integrate TreeSHAP for True Local Explainability
* Replace the heuristic `_factor_explanations()` with genuine SHAP waterfall plots:
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(features)
  ```
* Shows the exact positive and negative contribution of each feature to the prediction in real-time.

---

### Phase 3: Dashboard UI/UX & Feature Enhancements

#### 1. Interactive "What-If" Scenario Simulator Tab (implemented)
The dedicated tab starts with a selected real inspection context and accepts user-entered rainfall, wind, visibility, weather severity, water, debris, friction, surface condition, lighting, traffic, delays, flight operations, and previous incidents. The **Calculate risk** action returns the AIRS-RAN score, risk tier, component breakdown, model prediction, confidence, and recommendations. Scenarios are not persisted.

#### 2. Geospatial Airfield Canvas (Interactive Runway Diagram)
* Render an SVG/Canvas layout of the airport runways (e.g., Runway 07/25 and 12/30).
* Color-code runway segments dynamically according to current friction, water accumulation, and overall threat level.

#### 3. Automated Executive Alert Subsystem
* Configurable webhook/email alerts sending automated SMS or notification badges when any monitored runway breaches the $\ge 65.0$ risk threshold.
* Instant generation of formal PDF safety briefs for Air Traffic Control Tower supervisors.

---

## 4. Priority Implementation Matrix

| Task Item | Target Module | Complexity | Priority | Expected Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Fix Dynamic Weight Decrement Bug** | `prediction.py` | Low | **Immediate (P0)** | Eliminates counter-intuitive weight penalization during concurrent weather/traffic surges. |
| **Harmonize ML Label & AIRS-RAN Score** | `app.py` & `prediction.py` | Low | **Immediate (P0)** | Eliminates user confusion on Tab 2; provides clear unified guidance. |
| **Add "What-If" Simulation Playground** | `app.py` | Medium | **High (P1)** | Allows controllers to simulate future storms and forecast runway risk. |
| **Upgrade ML to XGBoost + SMOTE** | `ml_training.py` | Medium | **High (P1)** | Increases High-Risk recall from 58.8% to >85%. |
| **Integrate SHAP Explainability Plots**| `app.py` & `prediction.py` | Medium | **Medium (P2)** | Replaces static heuristics with verifiable game-theoretic AI explanations. |
| **Visual Airfield Segment Layout** | `app.py` | High | **Medium (P2)** | Delivers high-impact visual spatial awareness for airport tower operators. |
