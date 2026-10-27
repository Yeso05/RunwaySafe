from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).parent / "datasets"
DATASET_FILES = {
    "weather": ("weather_data.xlsx",),
    "inspections": ("runway_inspection_data.xlsx",),
    "flights": ("flight_operations_data.xlsx", "flight_operations_data (2).xlsx"),
    "incidents": ("historical_incidents_data.xlsx",),
}

TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET_NAMES = {value: key for key, value in TARGET_MAP.items()}


def _resolve_dataset(name: str) -> Path:
    for filename in DATASET_FILES[name]:
        path = DATASETS_DIR / filename
        if path.is_file():
            return path
    expected = ", ".join(DATASET_FILES[name])
    raise FileNotFoundError(f"Missing {name} dataset; expected one of: {expected}")


def load_datasets() -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_excel(_resolve_dataset(name), engine="openpyxl")
        for name in DATASET_FILES
    }
    required = {
        "weather": {
            "weather_id", "timestamp", "airport_code", "temperature_c",
            "rainfall_mm", "humidity_pct", "wind_speed_mps", "visibility_km",
            "pressure_hpa", "weather_severity",
        },
        "inspections": {
            "timestamp", "airport_code", "runway_id", "water_accumulation_mm",
            "debris_level", "surface_condition", "friction_coefficient",
            "lighting_status", "bird_activity_level", "inspector_status",
        },
        "flights": {
            "timestamp", "airport_code", "runway_assigned", "aircraft_type",
            "operation_type", "arrival_delay_min", "departure_delay_min",
            "traffic_density_pct", "weather_id",
        },
        "incidents": {
            "date", "airport_code", "runway_id", "risk_level",
            "incident_type", "weather_condition", "primary_cause",
            "risk_score", "outcome",
        },
    }
    for name, columns in required.items():
        missing = columns.difference(frames[name].columns)
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")
    frames["weather"]["timestamp"] = pd.to_datetime(
        frames["weather"]["timestamp"], errors="coerce"
    )
    frames["inspections"]["timestamp"] = pd.to_datetime(
        frames["inspections"]["timestamp"], errors="coerce"
    )
    frames["flights"]["timestamp"] = pd.to_datetime(
        frames["flights"]["timestamp"], errors="coerce"
    )
    frames["incidents"]["date"] = pd.to_datetime(
        frames["incidents"]["date"], errors="coerce"
    )
    if any(frame.isna().all(axis=None) for frame in frames.values()):
        raise ValueError("At least one dataset contains no usable values")
    return frames


def _daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    frame = weather.assign(date=weather["timestamp"].dt.normalize())
    numeric = [
        "temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_mps",
        "visibility_km", "pressure_hpa",
    ]
    result = frame.groupby(["airport_code", "date"], as_index=False)[numeric].mean()
    severity = (
        frame.groupby(["airport_code", "date"])["weather_severity"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else pd.NA)
        .reset_index()
    )
    return result.merge(severity, on=["airport_code", "date"], how="left")


def _daily_inspections(inspections: pd.DataFrame) -> pd.DataFrame:
    frame = inspections.assign(date=inspections["timestamp"].dt.normalize())
    numeric = [
        "water_accumulation_mm", "debris_level", "friction_coefficient",
        "bird_activity_level",
    ]
    result = frame.groupby(
        ["airport_code", "runway_id", "date"], as_index=False
    )[numeric].mean()
    for column in ["surface_condition", "lighting_status", "inspector_status"]:
        values = (
            frame.groupby(["airport_code", "runway_id", "date"])[column]
            .agg(lambda items: items.mode().iat[0] if not items.mode().empty else pd.NA)
            .reset_index()
        )
        result = result.merge(
            values, on=["airport_code", "runway_id", "date"], how="left"
        )
    return result


def _daily_flights(flights: pd.DataFrame) -> pd.DataFrame:
    frame = flights.assign(
        date=flights["timestamp"].dt.normalize(),
        runway_id=flights["runway_assigned"],
    )
    grouped = frame.groupby(
        ["airport_code", "runway_id", "date"], as_index=False
    ).agg(
        flight_count=("flight_id", "size"),
        arrival_delay_mean=("arrival_delay_min", "mean"),
        departure_delay_mean=("departure_delay_min", "mean"),
        traffic_density_mean=("traffic_density_pct", "mean"),
        aircraft_types=("aircraft_type", "nunique"),
        operation_types=("operation_type", "nunique"),
    )
    return grouped


def build_integrated_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one labeled row per historical incident without using label fields as features."""
    incidents = frames["incidents"].copy()
    incidents["date"] = incidents["date"].dt.normalize()
    unknown_levels = set(incidents["risk_level"].dropna()) - set(TARGET_MAP)
    if unknown_levels:
        raise ValueError(f"Unexpected incident risk levels: {sorted(unknown_levels)}")
    incidents["target"] = incidents["risk_level"].map(TARGET_MAP)
    incidents = incidents[incidents["target"].notna()].copy()
    incident_days = (
        incidents.groupby(["airport_code", "runway_id", "date"], as_index=False)
        .agg(
            day_incident_count=("incident_id", "size"),
            day_high_risk_count=("risk_level", lambda values: values.eq("High").sum()),
        )
        .sort_values(["airport_code", "runway_id", "date"])
    )
    day_group = incident_days.groupby(["airport_code", "runway_id"], sort=False)
    incident_days["prior_incident_count"] = day_group["day_incident_count"].cumsum() - incident_days["day_incident_count"]
    incident_days["prior_high_risk_count"] = day_group["day_high_risk_count"].cumsum() - incident_days["day_high_risk_count"]
    incidents = incidents.merge(
        incident_days[
            ["airport_code", "runway_id", "date", "prior_incident_count", "prior_high_risk_count"]
        ],
        on=["airport_code", "runway_id", "date"],
        how="left",
        validate="many_to_one",
    )

    result = incidents[
        ["incident_id", "airport_code", "runway_id", "date", "target",
         "prior_incident_count", "prior_high_risk_count"]
    ].merge(
        _daily_weather(frames["weather"]),
        on=["airport_code", "date"],
        how="left",
        validate="many_to_one",
    ).merge(
        _daily_inspections(frames["inspections"]),
        on=["airport_code", "runway_id", "date"],
        how="left",
        validate="many_to_one",
    ).merge(
        _daily_flights(frames["flights"]),
        on=["airport_code", "runway_id", "date"],
        how="left",
        validate="many_to_one",
    )
    return result


NUMERIC_FEATURES = [
    "temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_mps",
    "visibility_km", "pressure_hpa", "water_accumulation_mm", "debris_level",
    "friction_coefficient", "bird_activity_level", "flight_count",
    "arrival_delay_mean", "departure_delay_mean", "traffic_density_mean",
    "aircraft_types", "operation_types", "prior_incident_count",
    "prior_high_risk_count",
]
CATEGORICAL_FEATURES = [
    "airport_code", "runway_id", "weather_severity", "surface_condition",
    "lighting_status", "inspector_status",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
