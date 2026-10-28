# RunwaySafe: Technical Architecture, Preprocessing & ML Pipeline

## 1. Technical Pipeline Overview

RunwaySafe integrates multi-tabular aviation records, executes rigorous data pre-processing, trains supervised classification algorithms, and deploys the serialized model pipeline into a production-ready web decision-support application.

```
+--------------------+   +--------------------+   +--------------------+   +--------------------+
|  weather_data.xlsx |   | runway_inspection  |   | flight_operations  |   |historical_incidents|
+---------+----------+   +---------+----------+   +---------+----------+   +---------+----------+
          |                        |                        |                        |
          +------------------------+------------------------+------------------------+
                                   |
                                   v
             [ preprocessing.py : build_integrated_dataset() ]
             - Temporal normalization (date extraction)
             - Rolling cumulative past incidents (prevents future leakage)
             - Daily weather aggregations (mean & modal severity)
             - Airfield inspection merging (per runway & date)
             - Flight delays & density aggregations
                                   |
                                   v
             [ Scikit-Learn ColumnTransformer Pipeline ]
             +------------------------------+------------------------------+
             |       Numeric Pipeline       |     Categorical Pipeline     |
             | - Median Imputation          | - Most Frequent Imputation   |
             | - Missing Indicator Flags    | - One-Hot Encoding           |
             | - Standard Scaler (Z-score)  | - Unknown Category Handling  |
             +------------------------------+------------------------------+
                                   |
                                   v
             [ Model Training & Tournament : ml_training.py ]
             - DecisionTreeClassifier vs. RandomForestClassifier
             - Stratified 80/20 train/test split
             - Class weighting (handles class imbalance)
             - F1-Weighted tournament selection
                                   |
                                   v
             [ Model Artifact Generation ]
             - models/runwaysafe_model.joblib
             - results/evaluation_results.json
             - graphs/feature_importance.png
```

---

## 2. Data Preprocessing & Multi-Table Integration

The codebase contains raw data files across four operational domains: Weather, Runway Inspections, Flight Operations, and Historical Incidents.

### 2.1 File Resolution & Verification (`preprocessing.py`)
Data integrity checks ensure that all four Excel sheets exist and contain required schema definitions:

```python
# From preprocessing.py (lines 29-33)
def load_datasets() -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_excel(_resolve_dataset(name), engine="openpyxl")
        for name in DATASET_FILES
    }
```

The system verifies columns for each dataset:
* **Weather:** `temperature_c`, `rainfall_mm`, `humidity_pct`, `wind_speed_mps`, `visibility_km`, `pressure_hpa`, `weather_severity`.
* **Inspections:** `water_accumulation_mm`, `debris_level`, `surface_condition`, `friction_coefficient`, `lighting_status`, `bird_activity_level`, `inspector_status`.
* **Flights:** `aircraft_type`, `operation_type`, `arrival_delay_min`, `departure_delay_min`, `traffic_density_pct`.
* **Incidents:** `risk_level` (Target: Low=0, Medium=1, High=2), `date`, `airport_code`, `runway_id`.

### 2.2 Preventing Data Leakage in Historical Incidents
A major pitfall in predictive modeling is **target leakage** (using the future to predict the past). In `preprocessing.py`, the system calculates cumulative historical incidents strictly up to the day *prior* to each event:

```python
# From preprocessing.py (lines 148-150)
day_group = incident_days.groupby(["airport_code", "runway_id"], sort=False)
incident_days["prior_incident_count"] = (
    day_group["day_incident_count"].cumsum() - incident_days["day_incident_count"]
)
incident_days["prior_high_risk_count"] = (
    day_group["day_high_risk_count"].cumsum() - incident_days["day_high_risk_count"]
)
```
* Subtracting `day_incident_count` ensures that current-day incidents are **not counted as prior incidents**, maintaining strict temporal validity.

### 2.3 Aggregations and Cross-Domain Joins
Daily metrics are aggregated per airport and runway before joining:
* **Daily Weather:** Computes mean values for temperature, rain, wind, visibility, and extracts modal `weather_severity` (`_daily_weather()`, lines 77-89).
* **Daily Inspections:** Averages physical measurements and extracts dominant surface conditions (`_daily_inspections()`, lines 92-110).
* **Daily Flight Operations:** Aggregates total flights, average arrival/departure delays, and mean traffic density (`_daily_flights()`, lines 113-128).

---

## 3. Scikit-Learn Feature Transformation Pipeline

Data preprocessing is encapsulated inside a reproducible `ColumnTransformer` pipeline (`ml_training.py`):

```python
# From ml_training.py (lines 42-62)
def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
```

### Explaining the Pipeline Components:
1. **Median Imputer with Indicator:** If an sensor fails or a field is blank, the pipeline replaces it with the median value for that feature, but adds a boolean indicator column flag marking that the value was imputed.
2. **StandardScaler (Z-Score Normalization):**
   $$z = \frac{x - \mu}{\sigma}$$
   Ensures variables with large scales (e.g., pressure at ~1013 hPa) do not numerically overpower small scale variables (e.g., friction at ~0.65).
3. **OneHotEncoder (`handle_unknown='ignore'`):** Converts categorical strings (`"Good"`, `"Fair"`, `"Poor"`, airport codes) into binary vectors without throwing runtime errors on unseen categories in production.

---

## 4. Machine Learning Model Training & Tournament

### 4.1 Features Used
* **18 Numeric Features:** Temperature, rainfall, humidity, wind speed, visibility, pressure, water accumulation, debris level, friction coefficient, bird activity level, flight count, arrival delay mean, departure delay mean, traffic density mean, aircraft types count, operation types count, prior incident count, prior high risk count.
* **6 Categorical Features:** `airport_code`, `runway_id`, `weather_severity`, `surface_condition`, `lighting_status`, `inspector_status`.

### 4.2 Handling Class Imbalance
Real-world aviation safety data is naturally skewed—high-risk events are rare.
* Class Distribution: **Low (1,534)**, **Medium (178)**, **High (88)**.
* **Solution:** `class_weight="balanced"`. Both algorithms automatically penalize misclassifications of minority classes (High and Medium) inversely proportional to their frequencies:
  $$w_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$

### 4.3 Candidate Models & Hyperparameters

```python
# From ml_training.py (lines 129-137)
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, 
        class_weight="balanced", 
        max_depth=8
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, 
        random_state=42, 
        class_weight="balanced",
        n_jobs=-1, 
        min_samples_leaf=2
    ),
}
```

### 4.4 Automated Model Selection Tournament
The tournament evaluates both models on a held-out test split (20% of data, $N = 360$):

```python
# From ml_training.py (lines 147-154)
selected_name = max(
    evaluations,
    key=lambda name: (
        evaluations[name]["f1_weighted"],
        evaluations[name]["roc_auc_ovr_weighted"] or 0,
        evaluations[name]["accuracy"],
    ),
)
```

---

## 5. Model Evaluation & Benchmark Results

The training pipeline generates comprehensive test benchmarks stored in `results/evaluation_results.json`:

### Performance Comparison Matrix

| Evaluation Metric | Decision Tree | Random Forest (Selected Winner) |
| :--- | :--- | :--- |
| **Overall Accuracy** | 78.6% | **82.5%** |
| **Weighted Precision** | 83.5% | **82.9%** |
| **Weighted Recall** | 78.6% | **82.5%** |
| **Weighted F1-Score** | 80.6% | **82.7%** |
| **Multi-class ROC-AUC (OvR)**| 0.721 | **0.812** |

### Per-Class Detailed Breakdown (Random Forest)
* **Low Risk:** Precision = 91.1%, Recall = 90.2%, F1 = 90.7% (Support: 307)
* **Medium Risk:** Precision = 27.0%, Recall = 27.8%, F1 = 27.4% (Support: 36)
* **High Risk:** Precision = 52.6%, Recall = **58.8%**, F1 = 55.6% (Support: 17)

### Top 5 Most Influential Features (Random Forest Importance):
1. `numeric__water_accumulation_mm` (**18.5%**): Standing water depth is the single greatest predictor of runway hazard.
2. `numeric__rainfall_mm` (**10.4%**): Live precipitation intensity.
3. `numeric__friction_coefficient` (**9.3%**): Physical pavement deceleration grip.
4. `numeric__wind_speed_mps` (**5.0%**): Atmospheric gusts and crosswinds.
5. `numeric__departure_delay_mean` (**4.8%**): Airfield operational congestion.

---

## 6. Live Inference & Deployment Integration

In `prediction.py`, incoming requests are processed in real-time:
1. `_context_row()` extracts matching weather observations, flight queues, and inspection records.
2. The serialized pipeline `runwaysafe_model.joblib` transforms features and calls `.predict_proba()` to compute probability distributions across `[Low, Medium, High]`.
3. Concurrently, the AIRS-RAN deterministic algorithm computes the composite index ($0-100$).
4. Results are formatted with explainable factor contributions and delivered to the Streamlit UI in under 50 milliseconds.
