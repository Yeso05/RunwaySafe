# RunwaySafe: Mathematical Formulas & Calculation Methodologies

## 1. Overview

RunwaySafe combines **two distinct evaluation engines**:
1. **The Machine Learning Engine:** A trained Random Forest Classifier producing statistical class probabilities (`Low`, `Medium`, `High`) and confidence values.
2. **The AIRS-RAN Composite Index Engine:** A transparent, deterministic mathematical multi-criteria algorithm that outputs an exact risk score from `0 to 100` and dynamically re-allocates category weights based on active environmental threats.

This document details every mathematical equation, weighting table, normalization clamp, and aggregation formula used across the platform.

The dashboard also provides a manual What-if Risk Calculator. Its fields replace the corresponding values in one complete feature row, then the calculations below are applied without changing any source dataset.

---

## 2. AIRS-RAN Risk Component Calculations

All risk components are clamped within the range $[0.0, 100.0]$:
$$\text{Clamp}(x) = \max(0.0, \min(100.0, x))$$

### 2.1 Weather Risk Component ($R_{\text{weather}}$)
Measures the environmental hazard based on atmospheric restriction, precipitation, wind turbulence, and declared meteorological severity.

$$\begin{aligned}
R_{\text{weather}} = \text{Clamp}\Big( & \max(0.0, 10.0 - \text{visibility\_km}) \times 4.0 \\
& + \text{rainfall\_mm} \times 2.0 \\
& + \max(0.0, \text{wind\_speed\_mps} - 8.0) \times 3.0 \\
& + S_{\text{weather}} \Big)
\end{aligned}$$

Where $S_{\text{weather}}$ is a discrete severity surcharge:
$$S_{\text{weather}} = \begin{cases} 
20.0 & \text{if } \text{weather\_severity} = \text{"Severe"} \\
10.0 & \text{if } \text{weather\_severity} = \text{"Adverse"} \\
0.0 & \text{otherwise} 
\end{cases}$$

* **Logic Rationale:**
  * Standard visual range threshold is 10 km. Every kilometer below 10 km adds 4 points.
  * Every millimeter of rain adds 2 points.
  * Wind speeds up to 8 m/s (~16 knots) are standard; every 1 m/s above 8 m/s adds 3 points.

---

### 2.2 Runway Condition Risk Component ($R_{\text{runway}}$)
Measures the physical integrity and deceleration capability of the pavement surface.

$$\begin{aligned}
R_{\text{runway}} = \text{Clamp}\Big( & \text{water\_accumulation\_mm} \times 3.0 \\
& + \text{debris\_level} \times 10.0 \\
& + \max(0.0, 0.700 - \text{friction\_coefficient}) \times 100.0 \\
& + S_{\text{surface}} + S_{\text{lighting}} \Big)
\end{aligned}$$

Where:
$$S_{\text{surface}} = \begin{cases} 
18.0 & \text{if } \text{surface\_condition} = \text{"Poor"} \\
8.0 & \text{if } \text{surface\_condition} = \text{"Fair"} \\
0.0 & \text{if } \text{surface\_condition} = \text{"Good"}
\end{cases}$$

$$S_{\text{lighting}} = \begin{cases} 
12.0 & \text{if } \text{lighting\_status} = \text{"Degraded"} \\
0.0 & \text{otherwise}
\end{cases}$$

* **Logic Rationale:**
  * Standing water creates hydroplaning danger (scaled at 3 points per mm).
  * Safe runway friction benchmark is 0.700. If friction drops from 0.70 to 0.45, it adds $(0.70 - 0.45) \times 100 = 25$ risk points.

---

### 2.3 Flight Operations Risk Component ($R_{\text{operational}}$)
Quantifies runway congestion, operational tempo, and airline schedule stress.

$$\begin{aligned}
R_{\text{operational}} = \text{Clamp}\Big( & \text{traffic\_density\_mean} \times 0.45 \\
& + \max(0.0, \text{arrival\_delay\_mean}) \times 0.35 \\
& + \max(0.0, \text{departure\_delay\_mean}) \times 0.35 \\
& + \min(\text{flight\_count}, 100.0) \times 0.10 \Big)
\end{aligned}$$

* **Logic Rationale:**
  * High traffic density (% of hourly runway capacity) directly restricts go-around spacing and inspection windows.
  * Flight delays create cascading air traffic control pressure and pilot fatigue.

---

### 2.4 Historical Safety Risk Component ($R_{\text{historical}}$)
Evaluates past safety vulnerabilities on this specific runway.

$$R_{\text{historical}} = \text{Clamp}\Big( \min(\text{prior\_incident\_count} \times 3.0, 45.0) + \min(\text{prior\_high\_risk\_count} \times 8.0, 40.0) \Big)$$

* **Logic Rationale:**
  * Capped ceilings prevent historical incidents from dominating live conditions while ensuring chronic problem runways retain a higher baseline risk floor.

---

## 3. Dynamic Weighting & Composite AIRS-RAN Score

### 3.1 Baseline Weighting Matrix
In standard nominal operating conditions, the four categories contribute with fixed proportions:

| Risk Category | Baseline Weight ($w_i$) |
| :--- | :--- |
| **Runway Condition Risk** ($R_{\text{runway}}$) | **0.35** (35%) |
| **Weather Risk** ($R_{\text{weather}}$) | **0.25** (25%) |
| **Flight Operations Risk** ($R_{\text{operational}}$) | **0.25** (25%) |
| **Historical Safety Risk** ($R_{\text{historical}}$) | **0.15** (15%) |
| **Total Sum of Weights** | **1.00** (100%) |

### 3.2 Dynamic Weight Adaptation Rules
RunwaySafe adjusts weights dynamically in response to real-time triggers:

* **Rule 1 (Adverse Weather Shift):**
  If `weather_severity` is `"Adverse"` or `"Severe"`:
  $$\Delta w_{\text{weather}} = +0.10, \quad \Delta w_{\text{runway}} = -0.05, \quad \Delta w_{\text{operational}} = -0.05$$
  *(Weather weight increases from 0.25 to 0.35).*

* **Rule 2 (Congestion Surge Shift):**
  If `traffic_density_mean` $\ge 70.0\%$:
  $$\Delta w_{\text{operational}} = +0.10, \quad \Delta w_{\text{runway}} = -0.05, \quad \Delta w_{\text{historical}} = -0.05$$
  *(Operational weight increases from 0.25 to 0.35).*

### 3.3 Final Composite Risk Score Formulation
The final score $S_{\text{composite}}$ is calculated as:
$$S_{\text{composite}} = \text{Clamp}\left( \sum_{i=1}^{4} w_i \times R_i \right) = \text{Clamp}\Big( w_{\text{weather}} R_{\text{weather}} + w_{\text{runway}} R_{\text{runway}} + w_{\text{operational}} R_{\text{operational}} + w_{\text{historical}} R_{\text{historical}} \Big)$$

### 3.4 Risk Classification Tiering
The continuous score is mapped to operational risk categories:
$$\text{Risk Level} = \begin{cases} 
\text{High (🔴)} & \text{if } S_{\text{composite}} \ge 65.0 \\
\text{Medium (🟠)} & \text{if } 35.0 \le S_{\text{composite}} < 65.0 \\
\text{Low (🟢)} & \text{if } S_{\text{composite}} < 35.0
\end{cases}$$

---

## 4. Machine Learning Classification & Confidence Calculations

### 4.1 Ensemble Voting & Probability Distribution
The system uses an ensemble of $N = 300$ Decision Trees (Random Forest). For any feature vector $\mathbf{x}$, each tree $t$ outputs a leaf class distribution. The predicted probability for class $c \in \{\text{Low}(0), \text{Medium}(1), \text{High}(2)\}$ is:
$$P(Y = c \mid \mathbf{x}) = \frac{1}{N} \sum_{t=1}^{N} P_t(Y = c \mid \mathbf{x})$$

The categorical prediction $\hat{y}$ is selected via argmax:
$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} P(Y = c \mid \mathbf{x})$$

### 4.2 Prediction Confidence Score
The reported prediction confidence is the maximum class probability:
$$\text{Confidence} = \max_{c} P(Y = c \mid \mathbf{x})$$

For manual scenarios, the entered values are passed through the same saved preprocessing pipeline and classifier as historical context predictions. Missing operational fields that are not exposed as scenario controls remain populated from the selected inspection context.

---

## 5. Explainable AI: Feature Contribution Metric

To calculate the impact of each variable during an assessment, the system computes:
$$\text{Contribution}_j = I_j \times |z_j|$$
Where:
* $I_j$ is the global **Gini feature importance** of feature $j$ from the trained Random Forest model.
* $z_j$ is the **standardized feature value** after preprocessing ($z_j = \frac{x_j - \mu_j}{\sigma_j}$).

This highlights which features are driving the model's decision during the specific assessment context.
