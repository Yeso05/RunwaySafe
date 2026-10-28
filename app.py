from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from prediction import (
    available_contexts,
    latest_prediction,
    load_model_bundle,
    predict_context,
    predict_row,
    WEATHER_SEVERITY_VALUES,
)
from preprocessing import load_datasets

st.set_page_config(page_title="RunwaySafe", page_icon="✈️", layout="wide")
sns.set_theme(style="whitegrid")
ROOT = Path(__file__).parent


@st.cache_data
def get_datasets() -> dict[str, pd.DataFrame]:
    return load_datasets()


@st.cache_data
def cached_latest_prediction(airport: str, runway: str) -> dict:
    return latest_prediction(airport, runway)


def risk_badge(level: str) -> str:
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(level, "⚪")


def context_controls(
    datasets: dict[str, pd.DataFrame], key_prefix: str
) -> tuple[str, str, pd.Timestamp] | None:
    inspections = datasets["inspections"]
    airports = sorted(inspections["airport_code"].dropna().unique())
    airport = st.selectbox("Airport", airports, key=f"{key_prefix}_airport")
    runways = sorted(
        inspections.loc[inspections["airport_code"] == airport, "runway_id"]
        .dropna()
        .unique()
    )
    runway = st.selectbox("Runway", runways, key=f"{key_prefix}_runway")
    contexts = available_contexts(airport, runway)
    if contexts.empty:
        st.warning("No inspection contexts are available for this runway.")
        return None
    labels = [value.strftime("%Y-%m-%d %H:%M") for value in contexts["timestamp"]]
    selected = st.selectbox(
        "Available inspection date and time",
        labels,
        key=f"{key_prefix}_timestamp",
    )
    return airport, runway, pd.Timestamp(selected)


def show_prediction(result: dict) -> None:
    cols = st.columns(4)
    cols[0].metric(
        "Overall Runway Risk",
        f"{result['risk_score']:.0f}/100",
        f"{risk_badge(result['risk_level'])} {result['risk_level']}",
    )
    cols[1].metric("ML Prediction", result["label"])
    cols[2].metric("Prediction Confidence", f"{result['confidence']:.1%}")
    cols[3].metric("Assessment Time", result["timestamp"].strftime("%Y-%m-%d %H:%M"))
    st.subheader("AIRS-RAN risk components")
    st.caption(
        "AIRS-RAN is the proposed RunwaySafe decision-support framework. "
        "It is not a published aviation algorithm or autonomous authority."
    )
    component_frame = pd.DataFrame(
        {
            "Risk category": list(result["components"]),
            "Risk score": list(result["components"].values()),
            "Dynamic weight": [
                result["weights"][name] for name in result["components"]
            ],
        }
    ).set_index("Risk category")
    st.bar_chart(component_frame["Risk score"])
    st.dataframe(component_frame, width="stretch")
    st.subheader(f"Why is this runway considered {result['risk_level']} risk?")
    for factor in result["top_factors"]:
        st.write(f"- **{factor['factor']}**: {factor['value']}")
    st.subheader("Safety recommendations")
    for recommendation in result["recommendations"]:
        st.write(f"- {recommendation}")


def condition_table(result: dict) -> pd.DataFrame:
    row = result["row"]
    return pd.DataFrame(
        {
            "Condition": [
                "Rainfall",
                "Wind Speed",
                "Visibility",
                "Weather Severity",
                "Surface Condition",
                "Surface Water",
                "Runway Friction",
                "Traffic Density",
                "Flight Delay",
                "Previous Incidents",
            ],
            "Value": [
                f"{row['rainfall_mm']:.1f} mm",
                f"{row['wind_speed_mps']:.1f} m/s",
                f"{row['visibility_km']:.1f} km",
                row["weather_severity"] or "Unavailable",
                row["surface_condition"],
                f"{row['water_accumulation_mm']:.1f} mm",
                f"{row['friction_coefficient']:.3f}",
                f"{row['traffic_density_mean']:.1f}%",
                f"{row['arrival_delay_mean'] + row['departure_delay_mean']:.1f} min combined average",
                str(row["prior_incident_count"]),
            ],
        }
    ).assign(Value=lambda frame: frame["Value"].astype(str)).set_index("Condition")


def latest_assessments(inspections: pd.DataFrame) -> tuple[list[dict], list[str]]:
    assessments = []
    errors = []
    for airport in sorted(inspections["airport_code"].unique()):
        for runway in sorted(
            inspections.loc[inspections["airport_code"] == airport, "runway_id"].unique()
        ):
            try:
                result = cached_latest_prediction(airport, runway)
                result["airport"] = airport
                result["runway"] = runway
                assessments.append(result)
            except (FileNotFoundError, OSError, ValueError, KeyError) as error:
                errors.append(f"{airport}/{runway}: {error}")
    return assessments, errors


def alerts_for(result: dict) -> list[dict[str, str]]:
    row = result["row"]
    alerts = []
    checks = [
        ("Risk", result["risk_level"] == "High", "High", f"Overall AIRS-RAN risk is {result['risk_score']:.1f}/100."),
        ("Risk", result["risk_level"] == "Medium", "Medium", f"Overall AIRS-RAN risk is {result['risk_score']:.1f}/100."),
        ("Visibility", row["visibility_km"] < 5, "High", f"Visibility is {row['visibility_km']:.1f} km."),
        ("Surface water", row["water_accumulation_mm"] >= 5, "High", f"Surface water is {row['water_accumulation_mm']:.1f} mm."),
        ("Runway friction", row["friction_coefficient"] < 0.7, "High", f"Runway friction is {row['friction_coefficient']:.3f}."),
        ("Traffic density", row["traffic_density_mean"] >= 70, "Medium", f"Traffic density is {row['traffic_density_mean']:.1f}%."),
        ("Strong wind", row["wind_speed_mps"] >= 10, "Medium", f"Wind speed is {row['wind_speed_mps']:.1f} m/s."),
        ("Incident history", row["prior_high_risk_count"] > 0, "Medium", f"{row['prior_high_risk_count']} previous high-risk incidents are recorded."),
    ]
    for category, triggered, severity, reason in checks:
        if triggered:
            alerts.append({"Category": category, "Severity": severity, "Reason": reason})
    return alerts


def manual_assessment_form(
    baseline: dict,
) -> dict | None:
    row = baseline["row"]
    st.caption(
        "Enter the conditions you want to assess. Values are pre-filled from the "
        "selected inspection context and can be changed for a what-if scenario."
    )
    with st.form("manual_risk_calculator"):
        st.markdown("#### Weather conditions")
        weather_columns = st.columns(4)
        rainfall = weather_columns[0].number_input(
            "Rainfall (mm)", min_value=0.0, max_value=500.0,
            value=float(row["rainfall_mm"]), step=0.5,
            help="How much rain has fallen. More rain can add water and reduce safe runway performance, so higher values increase risk.",
        )
        wind_speed = weather_columns[1].number_input(
            "Wind speed (m/s)", min_value=0.0, max_value=100.0,
            value=float(row["wind_speed_mps"]), step=0.5,
            help="The speed of the wind near the runway. Stronger wind can make aircraft control harder, so higher values increase risk.",
        )
        visibility = weather_columns[2].number_input(
            "Visibility (km)", min_value=0.0, max_value=100.0,
            value=float(row["visibility_km"]), step=0.5,
            help="How far pilots can see. Shorter visibility makes landing and runway movement harder, so lower values increase risk.",
        )
        weather_severity = weather_columns[3].selectbox(
            "Weather severity", WEATHER_SEVERITY_VALUES,
            index=(
                WEATHER_SEVERITY_VALUES.index(row["weather_severity"])
                if row["weather_severity"] in WEATHER_SEVERITY_VALUES
                else 0
            ),
            help="A simple description of the overall weather. Adverse or Severe weather adds more weather risk than Normal.",
        )

        st.markdown("#### Runway surface")
        runway_columns = st.columns(4)
        water = runway_columns[0].number_input(
            "Surface water (mm)", min_value=0.0, max_value=100.0,
            value=float(row["water_accumulation_mm"]), step=0.5,
            help="The depth of water standing on the runway. More water can cause skidding or hydroplaning, so higher values increase risk.",
        )
        debris = runway_columns[1].number_input(
            "Debris level", min_value=0.0, max_value=10.0,
            value=float(row["debris_level"]), step=0.5,
            help="The amount of loose material such as stones or foreign objects. More debris can damage aircraft or reduce control, so higher values increase risk.",
        )
        friction = runway_columns[2].number_input(
            "Friction coefficient", min_value=0.0, max_value=1.0,
            value=float(row["friction_coefficient"]), step=0.01,
            help="How much grip the runway surface provides. Less grip means longer stopping distance, so lower values increase risk.",
        )
        surface = runway_columns[3].selectbox(
            "Surface condition", ["Good", "Fair", "Poor"],
            index=["Good", "Fair", "Poor"].index(
                row["surface_condition"]
                if row["surface_condition"] in {"Good", "Fair", "Poor"}
                else "Good"
            ),
            help="A visual assessment of pavement condition. Poor surfaces are more hazardous than Fair or Good surfaces.",
        )

        st.markdown("#### Operations and lighting")
        operation_columns = st.columns(4)
        lighting = operation_columns[0].selectbox(
            "Lighting status", ["Operational", "Degraded"],
            index=["Operational", "Degraded"].index(
                row["lighting_status"]
                if row["lighting_status"] in {"Operational", "Degraded"}
                else "Operational"
            ),
            help="Whether runway lighting is working normally. Degraded lighting can reduce pilot visibility, so Degraded increases risk.",
        )
        traffic = operation_columns[1].number_input(
            "Traffic density (%)", min_value=0.0, max_value=100.0,
            value=float(row["traffic_density_mean"]), step=1.0,
            help="How busy the runway is compared with its operating capacity. More traffic reduces spacing and inspection time, so higher values increase risk.",
        )
        arrival_delay = operation_columns[2].number_input(
            "Average arrival delay (min)", min_value=-120.0, max_value=500.0,
            value=float(row["arrival_delay_mean"]), step=1.0,
            help="The average delay for arriving flights. Positive delays indicate schedule pressure, so higher positive values increase operational risk.",
        )
        departure_delay = operation_columns[3].number_input(
            "Average departure delay (min)", min_value=-120.0, max_value=500.0,
            value=float(row["departure_delay_mean"]), step=1.0,
            help="The average delay for departing flights. Positive delays indicate schedule pressure, so higher positive values increase operational risk.",
        )

        st.markdown("#### Flight and incident history")
        history_columns = st.columns(3)
        flight_count = history_columns[0].number_input(
            "Flight operations", min_value=0, max_value=1000,
            value=int(row["flight_count"]), step=1,
            help="The number of flights using this runway in the assessment period. More movements increase operational workload, so higher values increase risk.",
        )
        prior_incidents = history_columns[1].number_input(
            "Previous incidents", min_value=0, max_value=1000,
            value=int(row["prior_incident_count"]), step=1,
            help="The number of earlier incidents recorded for this runway. A higher history indicates recurring concerns, so higher values increase risk.",
        )
        prior_high_risk = history_columns[2].number_input(
            "Previous high-risk incidents", min_value=0, max_value=1000,
            value=int(row["prior_high_risk_count"]), step=1,
            help="The number of earlier incidents classified as High risk. More serious history raises the safety baseline, so higher values increase risk.",
        )
        submitted = st.form_submit_button("Calculate risk", type="primary")

    if not submitted:
        return None
    manual_row = row.copy()
    manual_row.update(
        {
            "rainfall_mm": rainfall,
            "wind_speed_mps": wind_speed,
            "visibility_km": visibility,
            "weather_severity": weather_severity,
            "water_accumulation_mm": water,
            "debris_level": debris,
            "friction_coefficient": friction,
            "surface_condition": surface,
            "lighting_status": lighting,
            "traffic_density_mean": traffic,
            "arrival_delay_mean": arrival_delay,
            "departure_delay_mean": departure_delay,
            "flight_count": flight_count,
            "prior_incident_count": prior_incidents,
            "prior_high_risk_count": min(prior_high_risk, prior_incidents),
        }
    )
    return manual_row


try:
    datasets = get_datasets()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Unable to load the required datasets: {error}")
    st.stop()

weather = datasets["weather"]
inspections = datasets["inspections"]
flights = datasets["flights"]
incidents = datasets["incidents"]

st.title("RunwaySafe")
st.caption("Evidence-based runway risk assessment and operational decision support")

tabs = st.tabs(
    [
        "Home / Overview",
        "Runway Risk Assessment",
        "What-if Risk Calculator",
        "Current Conditions",
        "Safety Alerts",
        "Recommendations",
        "Risk History",
        "Analytics",
        "Reports",
    ]
)

with tabs[0]:
    st.header("Runway risk overview")
    assessments, assessment_errors = latest_assessments(inspections)
    high_risk = [item for item in assessments if item["risk_level"] == "High"]
    medium_risk = [item for item in assessments if item["risk_level"] == "Medium"]
    metrics = st.columns(4)
    metrics[0].metric("Total flights", f"{len(flights):,}")
    metrics[1].metric("Total inspections", f"{len(inspections):,}")
    metrics[2].metric("Total incidents", f"{len(incidents):,}")
    metrics[3].metric("Current/high-risk runways", f"{len(high_risk):,}")
    if assessments:
        overall_score = sum(item["risk_score"] for item in assessments) / len(assessments)
        overall_level = "High" if overall_score >= 65 else "Medium" if overall_score >= 35 else "Low"
        st.metric("Overall risk summary", f"{overall_score:.1f}/100", f"{risk_badge(overall_level)} {overall_level}")
        distribution = pd.Series(
            [item["risk_level"] for item in assessments], name="Runways"
        ).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.subheader("Current risk distribution")
        st.bar_chart(distribution)
        st.subheader("Important alerts")
        all_alerts = []
        for item in assessments:
            for alert in alerts_for(item):
                all_alerts.append(
                    {"Airport": item["airport"], "Runway": item["runway"], **alert}
                )
        if all_alerts:
            st.dataframe(pd.DataFrame(all_alerts), width="stretch", hide_index=True)
        else:
            st.success("No threshold-based alerts are active.")
    if assessment_errors:
        st.warning("Some runway assessments could not be calculated.")
        st.code("\n".join(assessment_errors))
    st.info("Use the Runway Risk Assessment tab to select a runway, inspect its real conditions, and generate a prediction.")

with tabs[1]:
    st.header("Runway Risk Assessment")
    selection = context_controls(datasets, "assessment")
    if selection:
        airport, runway, timestamp = selection
        try:
            result = predict_context(airport, runway, timestamp)
            st.subheader("Selected conditions")
            st.dataframe(condition_table(result), width="stretch")
            show_prediction(result)
            st.session_state["assessment_result"] = result
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"Unable to generate prediction: {error}")

with tabs[2]:
    st.header("What-if Risk Calculator")
    st.caption(
        "Use this calculator to test a proposed set of runway, weather, traffic, "
        "and history inputs. It does not change the stored datasets."
    )
    selection = context_controls(datasets, "manual")
    if selection:
        airport, runway, timestamp = selection
        try:
            baseline = predict_context(airport, runway, timestamp)
            manual_row = manual_assessment_form(baseline)
            if manual_row is not None:
                result = predict_row(manual_row, timestamp)
                st.subheader("Calculated risk")
                show_prediction(result)
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"Unable to calculate the requested scenario: {error}")

with tabs[3]:
    st.header("Current Conditions")
    selection = context_controls(datasets, "conditions")
    if selection:
        airport, runway, timestamp = selection
        try:
            result = predict_context(airport, runway, timestamp)
            st.metric(
                "Current risk indicator",
                f"{result['risk_score']:.0f}/100",
                f"{risk_badge(result['risk_level'])} {result['risk_level']}",
            )
            st.dataframe(condition_table(result), width="stretch")
            weather_day = result["supporting"]["weather"]
            flights_day = result["supporting"]["flights"]
            incidents_day = result["supporting"]["incidents"]
            st.caption(
                f"Context records: {len(weather_day):,} weather observations, "
                f"{len(flights_day):,} flight operations, {len(incidents_day):,} same-day incidents."
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"Unable to load current conditions: {error}")

with tabs[4]:
    st.header("Safety Alerts")
    st.caption("Thresholds are transparent rule-based signals derived from actual selected latest conditions.")
    assessments, errors = latest_assessments(inspections)
    alert_rows = []
    for item in assessments:
        for alert in alerts_for(item):
            alert_rows.append({"Airport": item["airport"], "Runway": item["runway"], **alert})
    if alert_rows:
        alert_frame = pd.DataFrame(alert_rows)
        st.metric("Active alerts", len(alert_frame))
        st.dataframe(alert_frame, width="stretch", hide_index=True)
    else:
        st.success("No high, medium, visibility, water, friction, traffic, wind, or incident alerts are active.")
    if errors:
        st.warning("Some alert assessments failed and require review.")
        st.code("\n".join(errors))

with tabs[5]:
    st.header("Recommendations")
    assessments, _ = latest_assessments(inspections)
    recommendations_rows = []
    for item in assessments:
        for recommendation in item["recommendations"]:
            recommendations_rows.append(
                {
                    "Airport": item["airport"],
                    "Runway": item["runway"],
                    "Risk": item["risk_level"],
                    "Recommendation": recommendation,
                }
            )
    if recommendations_rows:
        st.dataframe(pd.DataFrame(recommendations_rows), width="stretch", hide_index=True)
        st.caption("Recommendations are rule-based decision support from detected real conditions; they do not represent aviation certification or autonomous authority.")
    else:
        st.success("No additional recommendations were triggered.")

with tabs[6]:
    st.header("Risk History")
    history = incidents.copy()
    history["Date"] = history["date"].dt.date
    airports = sorted(history["airport_code"].unique())
    selected_airports = st.multiselect("Airports", airports, default=airports, key="history_airports")
    selected_levels = st.multiselect("Risk levels", ["Low", "Medium", "High"], default=["Low", "Medium", "High"], key="history_levels")
    filtered = history[
        history["airport_code"].isin(selected_airports)
        & history["risk_level"].isin(selected_levels)
    ]
    grouped = filtered.groupby(["Date", "airport_code", "runway_id", "risk_level"], as_index=False).agg(
        **{"Risk Score": ("risk_score", "mean"), "Incident Count": ("incident_id", "count")}
    )
    grouped["Risk Level"] = grouped.pop("risk_level")
    grouped = grouped.rename(columns={"airport_code": "Airport", "runway_id": "Runway"})
    st.dataframe(grouped.sort_values("Date", ascending=False).head(500), width="stretch", hide_index=True)
    trend = filtered.groupby(["date", "risk_level"]).size().unstack(fill_value=0)
    st.subheader("Incident risk trend")
    st.line_chart(trend)

with tabs[7]:
    st.header("Analytics")
    all_airports = sorted(set(weather["airport_code"]) | set(inspections["airport_code"]) | set(flights["airport_code"]) | set(incidents["airport_code"]))
    selected_airports = st.multiselect("Airports", all_airports, default=all_airports, key="analytics_airports")
    weather_view = weather[weather["airport_code"].isin(selected_airports)]
    inspections_view = inspections[inspections["airport_code"].isin(selected_airports)]
    flights_view = flights[flights["airport_code"].isin(selected_airports)]
    incidents_view = incidents[incidents["airport_code"].isin(selected_airports)]
    st.subheader("Weather Analytics")
    st.line_chart(weather_view.set_index("timestamp").resample("D")[["rainfall_mm", "wind_speed_mps", "visibility_km"]].mean())
    st.subheader("Runway Inspection Analytics")
    inspection_cols = st.columns(3)
    inspection_cols[0].metric("Average Surface Water", f"{inspections_view['water_accumulation_mm'].mean():.1f} mm")
    inspection_cols[1].metric("Average Runway Friction", f"{inspections_view['friction_coefficient'].mean():.3f}")
    inspection_cols[2].metric("Poor surface records", f"{(inspections_view['surface_condition'] == 'Poor').sum():,}")
    st.bar_chart(inspections_view["surface_condition"].value_counts())
    st.subheader("Flight Operations Analytics")
    st.bar_chart(flights_view["airport_code"].value_counts())
    st.line_chart(flights_view.set_index("timestamp")[["arrival_delay_min", "departure_delay_min"]].resample("D").mean())
    st.subheader("Incident Analytics")
    st.bar_chart(incidents_view["risk_level"].value_counts())
    st.bar_chart(incidents_view["incident_type"].value_counts())
    st.subheader("ML Feature Importance")
    evaluation_path = ROOT / "results" / "evaluation_results.json"
    if evaluation_path.is_file():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        importance = pd.DataFrame(evaluation["feature_importance"]).set_index("feature")
        st.bar_chart(importance.head(15)["importance"])
    else:
        st.warning("Evaluation results are not available. Run ml_training.py.")

with tabs[8]:
    st.header("Reports")
    st.caption("Generate a data-backed report for a selected inspection context and export it as CSV.")
    selection = context_controls(datasets, "report")
    if selection:
        airport, runway, timestamp = selection
        try:
            result = predict_context(airport, runway, timestamp)
            row = result["row"]
            report_rows = [
                ("Selected airport", airport),
                ("Selected runway", runway),
                ("Date/time", timestamp.strftime("%Y-%m-%d %H:%M")),
                ("Overall risk score", f"{result['risk_score']:.2f}/100"),
                ("Risk level", result["risk_level"]),
                ("ML prediction", result["label"]),
                ("Prediction confidence", f"{result['confidence']:.2%}"),
                ("Weather risk", f"{result['components']['Weather Risk']:.2f}"),
                ("Runway condition risk", f"{result['components']['Runway Condition Risk']:.2f}"),
                ("Operational risk", f"{result['components']['Flight Operations Risk']:.2f}"),
                ("Historical risk", f"{result['components']['Historical Safety Risk']:.2f}"),
                ("Top risk factors", "; ".join(f"{x['factor']}: {x['value']}" for x in result["top_factors"])),
                ("Recommendations", "; ".join(result["recommendations"])),
            ]
            evaluation_path = ROOT / "results" / "evaluation_results.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else {}
            report_rows.extend(
                [
                    ("Model", evaluation.get("selected_model", "Unavailable")),
                    ("Training samples", evaluation.get("training_samples", "Unavailable")),
                    ("Test samples", evaluation.get("test_samples", "Unavailable")),
                ]
            )
            report = pd.DataFrame(report_rows, columns=["Report item", "Value"])
            st.dataframe(report, width="stretch", hide_index=True)
            st.download_button(
                "Download runway safety report (CSV)",
                report.to_csv(index=False).encode("utf-8"),
                f"runwaysafe_{airport}_{runway.replace('/', '-')}_{timestamp:%Y%m%d_%H%M}.csv",
                "text/csv",
            )
            st.success("Report generated from the selected real dataset context and saved model.")
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"Unable to generate report: {error}")
