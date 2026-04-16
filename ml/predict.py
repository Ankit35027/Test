from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "simple_fleet_model.pkl"
_TOOLS = None
FEATURE_NAMES = [
    "usage_hours",
    "engine_temp",
    "tire_pressure",
    "oil_quality",
    "battery_voltage",
    "vibration_level",
    "maintenance_cost",
    "vehicle_type",
    "brake_condition",
]
NUMERICAL = [
    "usage_hours",
    "engine_temp",
    "tire_pressure",
    "oil_quality",
    "battery_voltage",
    "vibration_level",
    "maintenance_cost",
]


def _load_tools():
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = joblib.load(MODEL_PATH)
    return _TOOLS


def _normalize_vehicle_data(vehicle_data: dict) -> dict:
    """Accept either snake_case or app-style keys and map them to model features."""
    return {
        "usage_hours": vehicle_data.get("usage_hours", vehicle_data.get("Usage_Hours", 0)),
        "engine_temp": vehicle_data.get("engine_temp", vehicle_data.get("Engine_Temperature", 0.0)),
        "tire_pressure": vehicle_data.get("tire_pressure", vehicle_data.get("Tire_Pressure", 0.0)),
        "oil_quality": vehicle_data.get("oil_quality", vehicle_data.get("Oil_Quality", 0.0)),
        "battery_voltage": vehicle_data.get(
            "battery_voltage", vehicle_data.get("Battery_Voltage", 0.0)
        ),
        "vibration_level": vehicle_data.get(
            "vibration_level", vehicle_data.get("Vibration_Level", 0.0)
        ),
        "maintenance_cost": vehicle_data.get(
            "maintenance_cost", vehicle_data.get("Maintenance_Cost", 0.0)
        ),
        "vehicle_type": vehicle_data.get("vehicle_type", vehicle_data.get("Vehicle_Type", "Car")),
        "brake_condition": vehicle_data.get(
            "brake_condition", vehicle_data.get("Brake_Condition", "Good")
        ),
        "anomalies_detected": vehicle_data.get(
            "anomalies_detected",
            vehicle_data.get("Anomalies_Detected", 0),
        ),
        "failure_history": vehicle_data.get(
            "failure_history",
            vehicle_data.get("Failure_History", 0),
        ),
    }


def _build_contributing_factors(record: dict) -> dict:
    """Create an interpretable factor summary from the most relevant input readings."""
    factors = {}

    engine_temp = float(record["engine_temp"])
    oil_quality = float(record["oil_quality"])
    battery_voltage = float(record["battery_voltage"])
    tire_pressure = float(record["tire_pressure"])
    vibration_level = float(record["vibration_level"])
    brake_condition = str(record["brake_condition"])
    failure_history = int(record["failure_history"] in (1, True, "Yes", "yes"))
    anomalies_detected = int(record["anomalies_detected"] in (1, True, "Yes", "yes"))

    if engine_temp > 100:
        factors["engine_temperature"] = "Critical engine temperature"
    elif engine_temp >= 90:
        factors["engine_temperature"] = "Elevated engine temperature"

    if oil_quality < 0.3:
        factors["oil_quality"] = "Critical oil degradation"
    elif oil_quality < 0.6:
        factors["oil_quality"] = "Moderate oil wear"

    if battery_voltage < 11.5:
        factors["battery_voltage"] = "Battery failure risk"
    elif battery_voltage < 12.2:
        factors["battery_voltage"] = "Weak battery performance"

    if tire_pressure < 28:
        factors["tire_pressure"] = "Unsafe tire pressure"
    elif tire_pressure < 32:
        factors["tire_pressure"] = "Low tire pressure"

    if vibration_level > 0.8:
        factors["vibration_level"] = "Abnormal vibration detected"
    elif vibration_level >= 0.4:
        factors["vibration_level"] = "Moderate vibration trend"

    if brake_condition.lower() == "poor":
        factors["brake_condition"] = "Brake condition requires immediate repair"
    elif brake_condition.lower() == "fair":
        factors["brake_condition"] = "Brake servicing should be scheduled"

    if failure_history == 1:
        factors["failure_history"] = "Vehicle has previous failure history"

    if anomalies_detected == 1:
        factors["anomalies_detected"] = "Recent anomalies were detected"

    if not factors:
        return {"overall_status": "No major contributing factors detected from current inputs"}

    top_items = list(factors.items())[:3]
    return dict(top_items)


def _to_model_record(record: dict) -> dict:
    """Map the clean agent-facing record to the feature names used by the saved pipeline."""
    return {
        "Usage_Hours": record["usage_hours"],
        "Engine_Temperature": record["engine_temp"],
        "Tire_Pressure": record["tire_pressure"],
        "Oil_Quality": record["oil_quality"],
        "Battery_Voltage": record["battery_voltage"],
        "Vibration_Level": record["vibration_level"],
        "Maintenance_Cost": record["maintenance_cost"],
        "Anomalies_Detected": 1
        if record["anomalies_detected"] in (1, True, "Yes", "yes")
        else 0,
        "Failure_History": 1 if record["failure_history"] in (1, True, "Yes", "yes") else 0,
        "Vehicle_Type": record["vehicle_type"],
        "Brake_Condition": record["brake_condition"],
    }


def predict_maintenance(vehicle_data: dict) -> dict:
    """Run the saved ML pipeline and return a clean agent-compatible result dict."""
    tools = _load_tools()
    record = _normalize_vehicle_data(vehicle_data)
    model_record = _to_model_record(record)
    input_df = pd.DataFrame([model_record])

    input_num = input_df[tools["num_cols"]]
    input_cat = input_df[tools["cat_cols"]]

    num_filled = tools["num_imputer"].transform(input_num)
    cat_filled = tools["cat_imputer"].transform(input_cat)

    num_scaled = tools["scaler"].transform(num_filled)
    cat_encoded = tools["encoder"].transform(cat_filled)
    final_input = np.hstack((num_scaled, cat_encoded))

    risk_score = float(tools["model"].predict_proba(final_input)[0][1])
    risk_label = "High Risk" if risk_score >= 0.5 else "Safe"

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "contributing_factors": _build_contributing_factors(record),
    }
