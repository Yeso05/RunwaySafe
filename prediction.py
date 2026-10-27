from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_datasets

MODEL_PATH = Path(__file__).parent / "models" / "runwaysafe_model.joblib"

FRIENDLY_NAMES = {
    "rainfall_mm": "Rainfall",
    "wind_speed_mps": "Wind Speed",
    "visibility_km": "Visibility",
    "friction_coefficient": "Runway Friction",
    "water_accumulation_mm": "Surface Water",
    "traffic_density_mean": "Traffic Density",
    "flight_count": "Flight Operations",
    "arrival_delay_mean": "Arrival Delay",
    "departure_delay_mean": "Departure Delay",
    "prior_incident_count": "Previous Incidents",
    "prior_high_risk_count": "Previous High-Risk Incidents",
    "debris_level": "Debris Level",
    "bird_activity_level": "Bird Activity",
    "weather_severity": "Weather Severity",
    "surface_condition": "Surface Condition",
    "lighting_status": "Lighting",
}


def load_model_bundle() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run ml_training.py first."
        )
    return joblib.load(MODEL_PATH)


def available_contexts(airport_code: str, runway_id: str) -> pd.DataFrame:
    inspections = load_datasets()["inspections"]
    return (
        inspections[
            (inspections["airport_code"] == airport_code)
            & (inspections["runway_id"] == runway_id)
        ][["timestamp"]]
        .drop_duplicates()
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True)
    )


def _mode(frame: pd.DataFrame, column: str) -> str | None:
    values = frame[column].dropna().mode()
    return str(values.iat[0]) if not values.empty else None


def _context_row(
    frames: dict[str, pd.DataFrame], airport_code: str, runway_id: str, timestamp: pd.Timestamp
) -> tuple[dict, dict[str, object]]:
    weather = frames["weather"]
    inspections = frames["inspections"]
    flights = frames["flights"]
    incidents = frames["incidents"]
    date = timestamp.normalize()
    inspection = inspections[
        (inspections["airport_code"] == airport_code)
        & (inspections["runway_id"] == runway_id)
        & (inspections["timestamp"] == timestamp)
    ]
    if inspection.empty:
        raise ValueError("No inspection record exists for the selected context.")
    weather_day = weather[
        (weather["airport_code"] == airport_code)
        & (weather["timestamp"].dt.normalize() == date)
    ]
    flight_day = flights[
        (flights["airport_code"] == airport_code)
        & (flights["runway_assigned"] == runway_id)
        & (flights["timestamp"].dt.normalize() == date)
    ]
    incident_dates = pd.to_datetime(incidents["date"], errors="coerce")
    prior_incidents = incidents[
        (incidents["airport_code"] == airport_code)
        & (incidents["runway_id"] == runway_id)
        & (incident_dates < date)
    ]
    record = inspection.iloc[0]
    row = {
        "airport_code": airport_code,
        "runway_id": runway_id,
        "date": date,
        "temperature_c": weather_day["temperature_c"].mean(),
        "rainfall_mm": weather_day["rainfall_mm"].mean(),
        "humidity_pct": weather_day["humidity_pct"].mean(),
        "wind_speed_mps": weather_day["wind_speed_mps"].mean(),
        "visibility_km": weather_day["visibility_km"].mean(),
        "pressure_hpa": weather_day["pressure_hpa"].mean(),
        "weather_severity": _mode(weather_day, "weather_severity"),
        "water_accumulation_mm": record["water_accumulation_mm"],
        "debris_level": record["debris_level"],
        "friction_coefficient": record["friction_coefficient"],
        "bird_activity_level": record["bird_activity_level"],
        "surface_condition": record["surface_condition"],
        "lighting_status": record["lighting_status"],
        "inspector_status": record["inspector_status"],
        "flight_count": len(flight_day),
        "arrival_delay_mean": flight_day["arrival_delay_min"].mean(),
        "departure_delay_mean": flight_day["departure_delay_min"].mean(),
        "traffic_density_mean": flight_day["traffic_density_pct"].mean(),
        "aircraft_types": flight_day["aircraft_type"].nunique(),
        "operation_types": flight_day["operation_type"].nunique(),
        "prior_incident_count": len(prior_incidents),
        "prior_high_risk_count": int(prior_incidents["risk_level"].eq("High").sum()),
    }
    supporting = {
        "weather": weather_day,
        "inspection": inspection,
        "flights": flight_day,
        "incidents": incidents[
            (incidents["airport_code"] == airport_code)
            & (incidents["runway_id"] == runway_id)
            & (incident_dates == date)
        ],
    }
    return row, supporting


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _risk_components(row: dict) -> dict[str, float]:
    weather = (
        (max(0.0, 10.0 - float(row["visibility_km"])) * 4.0)
        + float(row["rainfall_mm"]) * 2.0
        + max(0.0, float(row["wind_speed_mps"]) - 8.0) * 3.0
        + (20.0 if row["weather_severity"] == "Severe" else 10.0 if row["weather_severity"] == "Adverse" else 0.0)
    )
    runway = (
        float(row["water_accumulation_mm"]) * 3.0
        + float(row["debris_level"]) * 10.0
        + max(0.0, 0.7 - float(row["friction_coefficient"])) * 100.0
        + (18.0 if row["surface_condition"] == "Poor" else 8.0 if row["surface_condition"] == "Fair" else 0.0)
        + (12.0 if row["lighting_status"] == "Degraded" else 0.0)
    )
    operational = (
        float(row["traffic_density_mean"]) * 0.45
        + max(0.0, float(row["arrival_delay_mean"])) * 0.35
        + max(0.0, float(row["departure_delay_mean"])) * 0.35
        + min(float(row["flight_count"]), 100.0) * 0.1
    )
    historical = (
        min(float(row["prior_incident_count"]) * 3.0, 45.0)
        + min(float(row["prior_high_risk_count"]) * 8.0, 40.0)
    )
    return {
        "Weather Risk": _clamp(weather),
        "Runway Condition Risk": _clamp(runway),
        "Flight Operations Risk": _clamp(operational),
        "Historical Safety Risk": _clamp(historical),
    }


def _airs_ran(components: dict[str, float], row: dict) -> tuple[float, dict[str, float]]:
    weights = {
        "Weather Risk": 0.25,
        "Runway Condition Risk": 0.35,
        "Flight Operations Risk": 0.25,
        "Historical Safety Risk": 0.15,
    }
    if row["weather_severity"] in {"Adverse", "Severe"}:
        weights["Weather Risk"] += 0.10
        weights["Runway Condition Risk"] -= 0.05
        weights["Flight Operations Risk"] -= 0.05
    if float(row["traffic_density_mean"]) >= 70:
        weights["Flight Operations Risk"] += 0.10
        weights["Runway Condition Risk"] -= 0.05
        weights["Historical Safety Risk"] -= 0.05
    score = sum(components[key] * value for key, value in weights.items())
    return _clamp(score), weights


def _factor_explanations(row: dict, components: dict[str, float]) -> list[dict[str, str]]:
    candidates = [
        ("Visibility", row["visibility_km"], f"{row['visibility_km']:.1f} km", max(0, 10 - row["visibility_km"]) * 4),
        ("Surface Water", row["water_accumulation_mm"], f"{row['water_accumulation_mm']:.1f} mm", row["water_accumulation_mm"] * 3),
        ("Runway Friction", row["friction_coefficient"], f"{row['friction_coefficient']:.3f}", max(0, 0.7 - row["friction_coefficient"]) * 100),
        ("Traffic Density", row["traffic_density_mean"], f"{row['traffic_density_mean']:.1f}%", row["traffic_density_mean"] * 0.45),
        ("Wind Speed", row["wind_speed_mps"], f"{row['wind_speed_mps']:.1f} m/s", max(0, row["wind_speed_mps"] - 8) * 3),
        ("Rainfall", row["rainfall_mm"], f"{row['rainfall_mm']:.1f} mm", row["rainfall_mm"] * 2),
        ("Previous Incidents", row["prior_incident_count"], str(row["prior_incident_count"]), row["prior_incident_count"] * 3),
    ]
    return [
        {"factor": name, "value": value, "reason": f"{name}: {value}", "score": f"{score:.1f}"}
        for name, _, value, score in sorted(candidates, key=lambda item: item[3], reverse=True)
        if score > 0
    ][:5]


def recommendations(row: dict) -> list[str]:
    actions = []
    if row["water_accumulation_mm"] >= 5 or row["rainfall_mm"] >= 5:
        actions.append("Monitor surface water and increase runway inspection frequency.")
    if row["friction_coefficient"] < 0.7 or row["surface_condition"] in {"Fair", "Poor"}:
        actions.append("Review runway friction and surface condition before continued operations.")
    if row["visibility_km"] < 5 or row["weather_severity"] in {"Adverse", "Severe"}:
        actions.append("Review weather conditions and alert safety/operations personnel.")
    if row["wind_speed_mps"] >= 10:
        actions.append("Monitor strong wind conditions and operational impacts.")
    if row["traffic_density_mean"] >= 70 or row["flight_count"] >= 60:
        actions.append("Monitor traffic density and coordinate with operations personnel.")
    if row["prior_incident_count"] > 0:
        actions.append("Review previous runway incidents and increase safety monitoring.")
    return actions or ["Continue routine monitoring and scheduled runway inspections."]


def predict_context(airport_code: str, runway_id: str, timestamp: pd.Timestamp) -> dict:
    row, supporting = _context_row(load_datasets(), airport_code, runway_id, timestamp)
    bundle = load_model_bundle()
    features = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    predicted = int(model.predict(features)[0])
    transformed = model.named_steps["preprocessor"].transform(features)
    classifier = model.named_steps["classifier"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    contributions = pd.DataFrame(
        {"feature": names, "importance": classifier.feature_importances_, "value": transformed[0]}
    )
    contributions["contribution"] = contributions["importance"] * contributions["value"].abs()
    components = _risk_components(row)
    score, weights = _airs_ran(components, row)
    level = "High" if score >= 65 else "Medium" if score >= 35 else "Low"
    return {
        "prediction": predicted,
        "label": bundle["target_names"][predicted],
        "probabilities": {
            bundle["target_names"][int(index)]: float(value)
            for index, value in enumerate(probabilities)
        },
        "confidence": float(probabilities[predicted]),
        "timestamp": timestamp,
        "row": row,
        "supporting": supporting,
        "risk_score": score,
        "risk_level": level,
        "components": components,
        "weights": weights,
        "top_factors": _factor_explanations(row, components),
        "recommendations": recommendations(row),
    }


def latest_prediction(airport_code: str, runway_id: str) -> dict:
    contexts = available_contexts(airport_code, runway_id)
    if contexts.empty:
        raise ValueError("No inspection record exists for the selected airport/runway.")
    return predict_context(airport_code, runway_id, contexts["timestamp"].iat[0])
