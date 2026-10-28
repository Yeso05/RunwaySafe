# RunwaySafe Dashboard: Unboxing & User Walkthrough

Welcome to the **RunwaySafe User Dashboard**! 

Think of this document like an **unboxing guide for a brand-new high-tech instrument**. We will take you on a complete visual and functional tour across every screen, button, metric card, and table in the application. You will learn exactly where every element is located on the screen, what every term means in plain airport language, and how to use the dashboard during daily airport operations.

---

## Screen Architecture Overview

When you launch RunwaySafe, you see a light navigation interface with the application title **"RunwaySafe"** and a series of **9 distinct functional tabs**:

The dashboard uses a professional aviation-safety palette: a light blue-grey page background, white panels, aviation blue navigation, teal accents, and semantically consistent green, amber, and red risk colors. Charts and tables use the same light theme for readability.

```
+-----------------------------------------------------------------------------------------------------+
| ✈️ RunwaySafe                                                                                        |
| Evidence-based runway risk assessment and operational decision support                              |
+-----------------------------------------------------------------------------------------------------+
| [Tab 3] What-if Calculator | [Tab 4] Conditions | [Tab 5] Safety Alerts | [Tab 6] Recommendations |
| [Tab 7] Risk History | [Tab 8] Analytics | [Tab 9] Reports                                      |
+-----------------------------------------------------------------------------------------------------+
```

---

## Tab 1: "Home / Overview"

### Purpose
The **Command Center**. This is the landing screen designed for airport directors and duty managers who need a 5-second pulse check on the entire airport's runway status.

```
+-----------------------------------------------------------------------------------------------------+
| Top KPI Metrics:                                                                                    |
| [ Total Flights: 12,450 ] [ Total Inspections: 820 ] [ Total Incidents: 1,800 ] [ High-Risk RWYs: 2]|
+-----------------------------------------------------------------------------------------------------+
| Overall Risk Summary:  [ 42.6 / 100 ]  🟠 Medium                                                    |
+-----------------------------------------------------------------------------------------------------+
| Left Side: [Current Risk Distribution (Bar Chart)] | Right Side: [Important Active Alerts (Table)]  |
| - Low:    4 Runways                                | - VOMM / 07: Surface water 6.2mm (High)        |
| - Medium: 3 Runways                                | - VIDP / 28: Runway friction 0.580 (High)      |
| - High:   2 Runways                                |                                                |
+-----------------------------------------------------------------------------------------------------+
```

### Component Breakdown & Terminology

1. **Top KPI Metric Cards (Top Row of Tab 1):**
   * **Total flights:** The total number of flight operations recorded across the integrated dataset.
   * **Total inspections:** The count of physical airfield safety runway inspections recorded.
   * **Total incidents:** Historical database count of safety incidents, bird strikes, foreign object debris reports, or runway excursions.
   * **Current/high-risk runways:** The count of runways currently operating under a **High Risk (🔴)** classification across all monitored airports.

2. **Overall Risk Summary Metric (Directly below Top Cards):**
   * Displays the fleet-wide or airport-wide average composite risk score (0 to 100).
   * **Badges:**
     * 🟢 **Low:** Score `< 35` — Normal routine runway operations.
     * 🟠 **Medium:** Score `35 to 64` — Elevated caution; active advisory monitoring.
     * 🔴 **High:** Score `≥ 65` — Critical hazard; immediate intervention required.

3. **Current Risk Distribution Chart:**
   * A clean bar graph showing how many monitored runways currently fall into **Low**, **Medium**, and **High** risk brackets.

4. **Important Alerts Table:**
   * Summarizes all active threshold breaches across the runways (e.g., if friction has dropped below standard or water is ponding).

---

## Tab 2: "Runway Risk Assessment"

### Purpose
The **Deep-Dive Diagnostic Station**. When an inspector or air traffic controller wants to investigate a specific runway at a specific time, this is where the AI and mathematical models generate a detailed safety profile.

```
+-----------------------------------------------------------------------------------------------------+
| Selection Controls:                                                                                 |
| Airport: [ VOMM (Chennai) v ]   Runway: [ 07/25 v ]   Inspection Timestamp: [ 2024-03-15 14:30 v ]  |
+-----------------------------------------------------------------------------------------------------+
| Selected Conditions (Table of Real Field Measurements):                                             |
| Rainfall: 12.4 mm | Wind: 9.2 m/s | Visibility: 4.1 km | Surface: Wet | Friction: 0.580 ...         |
+-----------------------------------------------------------------------------------------------------+
| Risk Diagnostic Metrics (4 Cards):                                                                  |
| [ Overall Runway Risk ] [ ML Prediction ] [ Prediction Confidence ] [ Assessment Timestamp ]       |
|      68 / 100  🔴 High        High Risk                 88.4%              2024-03-15 14:30         |
+-----------------------------------------------------------------------------------------------------+
| AIRS-RAN Risk Components Breakdown (Bar Chart & Table):                                             |
| - Weather Risk:           58.2  (Dynamic Weight: 35%)                                               |
| - Runway Condition Risk:  74.5  (Dynamic Weight: 30%)                                               |
| - Flight Operations Risk: 62.0  (Dynamic Weight: 25%)                                               |
| - Historical Safety Risk: 45.0  (Dynamic Weight: 10%)                                               |
+-----------------------------------------------------------------------------------------------------+
| Why is this runway considered High risk? (Top Contributing Factors):                                |
| • Surface Water: 6.2 mm ponding depth                                                               |
| • Runway Friction: 0.580 (below safety reference of 0.700)                                          |
| • Visibility: 4.1 km (restricted visibility below 10 km)                                            |
+-----------------------------------------------------------------------------------------------------+
| Safety Recommendations:                                                                             |
| • Monitor surface water and increase runway inspection frequency.                                   |
| • Review runway friction and surface condition before continued operations.                         |
+-----------------------------------------------------------------------------------------------------+
```

### Component Breakdown & Terminology

1. **Context Selector Bar (Top of Tab 2):**
   * **Airport Dropdown:** Select the target airport by ICAO code (e.g., `VOMM` for Chennai, `VIDP` for Delhi).
   * **Runway Dropdown:** Select the physical runway designator (e.g., `07/25`, `12/30`).
   * **Inspection Date & Time Dropdown:** Select the exact inspection log event to examine.

2. **Selected Conditions Table:**
   * A clean inventory showing the physical environmental and operational measurements recorded during that inspection window.

3. **Core Diagnostic Metrics (4-Column Card Grid):**
   * **Overall Runway Risk:** The composite AIRS-RAN index score on a scale of `0 to 100` alongside its color-coded badge.
   * **ML Prediction:** The Machine Learning model's categorical classification: `Low`, `Medium`, or `High`.
   * **Prediction Confidence:** The mathematical certainty of the AI model expressed as a percentage (e.g., `88.4%`).
   * **Assessment Time:** The local timestamp corresponding to the inspection record.

4. **AIRS-RAN Risk Components:**
   * **AIRS-RAN:** Acronym for *Aviation Integrated Runway Safety - Risk Assessment Network*. It is RunwaySafe's structured multi-criteria decision-support framework.
   * **Component Breakdown Bar Chart:** Visually separates risk into 4 independent sub-scores:
     * *Weather Risk:* How hostile the current atmosphere is (wind, rain, visibility).
     * *Runway Condition Risk:* How compromised the physical pavement is (water depth, rubber/friction loss, debris, lighting).
     * *Flight Operations Risk:* The operational strain on the runway (traffic density %, arrival/departure delays, flight frequency).
     * *Historical Safety Risk:* The baseline vulnerability of this specific runway based on past incident counts.
   * **Dynamic Weight:** The percentage influence given to each component. Under severe weather or heavy flight congestion, the system dynamically shifts weights to prioritize the active hazard.

5. **Top Risk Factors ("Why is this runway considered X risk?"):**
   * Directly explains to the human operator which factors exerted the strongest upward pressure on the risk score, eliminating algorithmic opacity.

6. **Safety Recommendations:**
   * Rule-based actionable checklists alerting the airfield operations supervisor to dispatch sweeping vehicles, friction test trailers, or coordinate air traffic spacing.

---

## Tab 3: "What-if Risk Calculator"

### Purpose
The What-if Risk Calculator lets a duty manager enter a proposed set of weather, runway, traffic, and incident-history values and immediately calculate the resulting risk. It is a scenario tool only: it does not modify the source datasets or historical records.

1. Select an airport, runway, and inspection context. The context provides sensible starting values.
2. Edit rainfall, wind, visibility, weather severity, surface water, debris, friction, surface condition, lighting, traffic density, delays, flight operations, and previous incidents.
3. Edit previous incidents and previous high-risk incidents independently, then select **Calculate risk** to run both the AIRS-RAN score and the saved Random Forest model against the entered row.
4. Review the clearly labelled Overall Risk Score /100, Risk Level, ML Prediction, Confidence, Top Risk Factors, and Recommended Actions, followed by component scores and dynamic weights.

The calculator uses the same thresholds as the standard assessment: Low below 35, Medium from 35 through 64.99, and High at 65 or above. The ML prediction is an historical baseline; the AIRS-RAN score is the transparent current-condition indicator.

## Tab 4: "Current Conditions"

### Purpose
The **Field Observation Viewer**. Designed for airfield inspectors who need to see the raw physical parameters measured on the tarmac along with contextual data counts.

```
+-----------------------------------------------------------------------------------------------------+
| Airport & Runway Selector: [ VOMM v ] [ Runway 07 v ] [ Timestamp: Latest v ]                       |
+-----------------------------------------------------------------------------------------------------+
| Current Risk Indicator: [ 68/100 ] 🔴 High                                                          |
| Comprehensive Conditions Table (Physical, Chemical, & Operational attributes)                       |
| Context Records Footer: 24 weather observations, 48 flight operations, 1 same-day incident          |
+-----------------------------------------------------------------------------------------------------+
```

### Key Parameters Explained:
* **Water Accumulation (mm):** Depth of standing water on the runway surface. Depths exceeding 3mm pose hydroplaning risks for landing aircraft.
* **Friction Coefficient:** Dimensionless continuous reading (typically between 0.20 and 0.85). Values below 0.70 represent reduced braking action.
* **Debris Level:** Quantitative measure of loose gravel, rubber chunks, or Foreign Object Debris (FOD) on the tarmac.
* **Lighting Status:** Operational state of the runway center-line, touchdown zone, and edge lights (`Normal`, `Degraded`, `Failed`).
* **Traffic Density (%):** Percentage utilization of the runway capacity relative to standard hourly throughput.

---

## Tab 5: "Safety Alerts"

### Purpose
The **Active Warning Board**. Collects and highlights all active safety threshold violations across all monitored runways in the airport network.

```
+-----------------------------------------------------------------------------------------------------+
| Active Alerts Counter: [ 7 Active Alerts Across 3 Runways ]                                         |
+-----------------------------------------------------------------------------------------------------+
| Table of Active Alerts:                                                                             |
| Airport | Runway | Category        | Severity  | Reason                                             |
|---------+--------+-----------------+-----------+----------------------------------------------------|
| VOMM    | 07     | Surface Water   | High      | Surface water is 6.2 mm (Threshold >= 5.0 mm)      |
| VOMM    | 07     | Runway Friction | High      | Runway friction is 0.580 (Threshold < 0.700)       |
| VIDP    | 28     | Visibility      | High      | Visibility is 3.8 km (Threshold < 5.0 km)          |
| VIDP    | 28     | Traffic Density | Medium    | Traffic density is 74.2% (Threshold >= 70.0%)      |
+-----------------------------------------------------------------------------------------------------+
```

### How to Interpret Alert Categories:
* **Surface Water (High Severity):** Water accumulation $\ge 5.0\text{ mm}$ — High hydroplaning hazard.
* **Runway Friction (High Severity):** Friction coefficient $< 0.700$ — Impaired aircraft deceleration.
* **Visibility (High Severity):** Meteorological visibility $< 5.0\text{ km}$ — Instrument landing restrictions apply.
* **Strong Wind (Medium Severity):** Wind speed $\ge 10.0\text{ m/s}$ ($\approx 20\text{ knots}$) — Crosswind and shear watch.
* **Traffic Density (Medium Severity):** Traffic density $\ge 70.0\%$ — Runway capacity saturation risk.
* **Incident History (Medium Severity):** Runway has recorded prior high-risk incidents — heightened monitoring protocol.

---

## Tab 6: "Recommendations"

### Purpose
The **Operational Advisory Feed**. Aggregates all automated safety advisories generated across the airport system into a single dispatchable table for operations teams.

* **Example Action Items:**
  * *"Monitor surface water and increase runway inspection frequency."*
  * *"Review runway friction and surface condition before continued operations."*
  * *"Monitor traffic density and coordinate with air traffic management."*
  * *"Continue routine monitoring and scheduled runway inspections."*

---

## Tab 7: "Risk History"

### Purpose
The **Historical Incident & Trend Archive**. Used by airport safety auditors and safety directors to inspect historical incident patterns over time.

```
+-----------------------------------------------------------------------------------------------------+
| Filters: Airport Multiselect [ VOMM x, VIDP x ]   Risk Level Multiselect [ High x, Medium x ]       |
+-----------------------------------------------------------------------------------------------------+
| Incident Record History (Table of prior logged events with dates, scores, and counts)               |
+-----------------------------------------------------------------------------------------------------+
| Incident Risk Trend (Time-Series Line Chart):                                                       |
| Shows chronological frequency of Low, Medium, and High risk incidents across weeks and months       |
+-----------------------------------------------------------------------------------------------------+
```

---

## Tab 8: "Analytics"

### Purpose
The **Cross-Domain Business Intelligence Studio**. Breaks down the runway ecosystem into 4 analytical quadrants:
1. **Weather Analytics:** Multi-line graph of daily rainfall, wind speed, and visibility trends over time.
2. **Runway Inspection Analytics:** Metric cards showing average water accumulation, average friction coefficient, counts of "Poor" surface conditions, and distribution bar charts.
3. **Flight Operations Analytics:** Aircraft traffic volume distribution by airport and daily arrival/departure delay trends.
4. **Incident Analytics & ML Feature Importance:** Breakdown of incident types (bird strikes, excursions, FOD) and the **Machine Learning Feature Importance bar chart** showing which physical factors most heavily influence the AI's risk predictions.

---

## Tab 9: "Reports"

### Purpose
The **Compliance & Export Station**. Allows users to select any specific runway inspection context, view a formal compliance summary, and click:
* **[Download runway safety report (CSV)]**: Exports an official, timestamped safety assessment file ready to be archived, emailed to airport directors, or submitted to civil aviation regulators (e.g., FAA, DGCA, EASA).
