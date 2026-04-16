from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Training_Data" / "fleet_maintenance_unbiased_40k.csv"
MODEL_PATH = BASE_DIR / "simple_fleet_model.pkl"

NUM_COLS = [
    "Usage_Hours",
    "Engine_Temperature",
    "Tire_Pressure",
    "Oil_Quality",
    "Battery_Voltage",
    "Vibration_Level",
    "Maintenance_Cost",
    "Anomalies_Detected",
    "Failure_History",
]
CAT_COLS = ["Vehicle_Type", "Brake_Condition"]
TARGET_COL = "Maintenance_Required"


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("°F", "", regex=False)
        .str.replace("°C", "", regex=False)
        .str.strip()
        .replace({"Missing": None, "nan": None, "None": None, "": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_brake_condition(series: pd.Series) -> pd.Series:
    mapping = {
        "g": "Good",
        "good": "Good",
        "gud": "Good",
        "ok": "Fair",
        "f": "Fair",
        "fair": "Fair",
        "p": "Poor",
        "poor": "Poor",
        "bad": "Poor",
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna("Fair")
    )


def _convert_temperature_to_celsius(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    fahrenheit_mask = raw.str.contains("°F", regex=False, na=False)
    numeric = _to_float(raw)
    numeric.loc[fahrenheit_mask] = (numeric.loc[fahrenheit_mask] - 32.0) * (5.0 / 9.0)
    return numeric


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["Usage_Hours"] = _to_float(df["Usage_Hours"])
    df["Engine_Temperature"] = _convert_temperature_to_celsius(df["Engine_Temperature"])
    df["Tire_Pressure"] = _to_float(df["Tire_Pressure"])
    df["Oil_Quality"] = _to_float(df["Oil_Quality"]) / 100.0
    df["Battery_Voltage"] = _to_float(df["Battery_Voltage"])
    df["Vibration_Level"] = _to_float(df["Vibration_Level"])
    df["Maintenance_Cost"] = _to_float(df["Maintenance_Cost"])

    df["Vehicle_Type"] = df["Vehicle_Type"].astype(str).str.strip().fillna("Car")
    df["Brake_Condition"] = _normalize_brake_condition(df["Brake_Condition"])
    df["Anomalies_Detected"] = pd.to_numeric(df["Anomalies_Detected"], errors="coerce").fillna(0)
    df["Failure_History"] = pd.to_numeric(df["Failure_History"], errors="coerce").fillna(0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    return df


def rebuild_model() -> None:
    df = load_training_frame()

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].copy()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = num_imputer.fit_transform(X[NUM_COLS])
    X_cat = cat_imputer.fit_transform(X[CAT_COLS])

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_num_scaled = scaler.fit_transform(X_num)
    X_cat_encoded = encoder.fit_transform(X_cat)

    X_final = pd.concat(
        [
            pd.DataFrame(X_num_scaled),
            pd.DataFrame(X_cat_encoded),
        ],
        axis=1,
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_final, y)

    payload = {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved rebuilt model to {MODEL_PATH}")


if __name__ == "__main__":
    rebuild_model()
