from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Training_Data" / "fleet_maintenance_unbiased_40k.csv"
MODEL_PATH = BASE_DIR / "simple_fleet_model.pkl"
GUIDELINES_PATH = BASE_DIR / "rag" / "guidelines" / "maintenance_manual.txt"
VECTORSTORE_PATH = BASE_DIR / "rag" / "vectorstore"

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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

_TOOLS = None
_VECTORSTORE = None


class FleetState(TypedDict):
    vehicle_data: dict
    risk_score: float
    risk_label: str
    contributing_factors: dict
    retrieved_guidelines: str
    health_summary: str
    action_plan: str
    disclaimer: str


def _build_embeddings() -> HuggingFaceEmbeddings:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"token": token} if token else {}
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)


def build_vectorstore() -> None:
    loader = TextLoader(str(GUIDELINES_PATH))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, _build_embeddings())
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))


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
    return series.astype(str).str.strip().str.lower().map(mapping).fillna("Fair")


def _convert_temperature_to_celsius(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    fahrenheit_mask = raw.str.contains("°F", regex=False, na=False)
    numeric = _to_float(raw)
    numeric.loc[fahrenheit_mask] = (numeric.loc[fahrenheit_mask] - 32.0) * (5.0 / 9.0)
    return numeric


def rebuild_model() -> None:
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
    X_final = np.hstack((X_num_scaled, X_cat_encoded))

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


def _load_tools():
    global _TOOLS
    if _TOOLS is None:
        try:
            _TOOLS = joblib.load(MODEL_PATH)
        except Exception:
            rebuild_model()
            _TOOLS = joblib.load(MODEL_PATH)
    return _TOOLS


def _normalize_vehicle_data(vehicle_data: dict) -> dict:
    return {
        "usage_hours": vehicle_data.get("usage_hours", vehicle_data.get("Usage_Hours", 0)),
        "engine_temp": vehicle_data.get("engine_temp", vehicle_data.get("Engine_Temperature", 0.0)),
        "tire_pressure": vehicle_data.get("tire_pressure", vehicle_data.get("Tire_Pressure", 0.0)),
        "oil_quality": vehicle_data.get("oil_quality", vehicle_data.get("Oil_Quality", 0.0)),
        "battery_voltage": vehicle_data.get("battery_voltage", vehicle_data.get("Battery_Voltage", 0.0)),
        "vibration_level": vehicle_data.get("vibration_level", vehicle_data.get("Vibration_Level", 0.0)),
        "maintenance_cost": vehicle_data.get("maintenance_cost", vehicle_data.get("Maintenance_Cost", 0.0)),
        "vehicle_type": vehicle_data.get("vehicle_type", vehicle_data.get("Vehicle_Type", "Car")),
        "brake_condition": vehicle_data.get("brake_condition", vehicle_data.get("Brake_Condition", "Good")),
        "anomalies_detected": vehicle_data.get("anomalies_detected", vehicle_data.get("Anomalies_Detected", 0)),
        "failure_history": vehicle_data.get("failure_history", vehicle_data.get("Failure_History", 0)),
        "vehicle_id": vehicle_data.get("vehicle_id", "VH-001"),
    }


def _build_contributing_factors(record: dict) -> dict:
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
    return dict(list(factors.items())[:3])


def predict_maintenance(vehicle_data: dict) -> dict:
    tools = _load_tools()
    record = _normalize_vehicle_data(vehicle_data)
    model_record = {
        "Usage_Hours": record["usage_hours"],
        "Engine_Temperature": record["engine_temp"],
        "Tire_Pressure": record["tire_pressure"],
        "Oil_Quality": record["oil_quality"],
        "Battery_Voltage": record["battery_voltage"],
        "Vibration_Level": record["vibration_level"],
        "Maintenance_Cost": record["maintenance_cost"],
        "Anomalies_Detected": 1 if record["anomalies_detected"] in (1, True, "Yes", "yes") else 0,
        "Failure_History": 1 if record["failure_history"] in (1, True, "Yes", "yes") else 0,
        "Vehicle_Type": record["vehicle_type"],
        "Brake_Condition": record["brake_condition"],
    }

    input_df = pd.DataFrame([model_record])
    input_num = input_df[tools["num_cols"]]
    input_cat = input_df[tools["cat_cols"]]
    num_filled = tools["num_imputer"].transform(input_num)
    cat_filled = tools["cat_imputer"].transform(input_cat)
    num_scaled = tools["scaler"].transform(num_filled)
    cat_encoded = tools["encoder"].transform(cat_filled)
    final_input = np.hstack((num_scaled, cat_encoded))

    risk_score = float(tools["model"].predict_proba(final_input)[0][1])
    return {
        "risk_score": risk_score,
        "risk_label": "High Risk" if risk_score >= 0.5 else "Safe",
        "contributing_factors": _build_contributing_factors(record),
    }


def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        if not VECTORSTORE_PATH.exists():
            build_vectorstore()
        _VECTORSTORE = FAISS.load_local(
            str(VECTORSTORE_PATH),
            _build_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _VECTORSTORE


def retrieve_guidelines(query: str) -> str:
    retriever = _get_vectorstore().as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def _extract_section(text: str, section_name: str) -> str:
    try:
        start = text.index(f"{section_name}:") + len(f"{section_name}:")
        next_sections = ["HEALTH SUMMARY:", "ACTION PLAN:", "DISCLAIMER:"]
        end = len(text)
        for section in next_sections:
            if section != f"{section_name}:" and section in text[start:]:
                end = min(end, text.index(section, start))
        return text[start:end].strip()
    except ValueError:
        return "Not available."


def run_fleet_agent(vehicle_data: dict) -> FleetState:
    state: FleetState = {
        "vehicle_data": vehicle_data,
        "risk_score": 0.0,
        "risk_label": "Unknown",
        "contributing_factors": {},
        "retrieved_guidelines": "",
        "health_summary": "",
        "action_plan": "",
        "disclaimer": "",
    }

    ml_result = predict_maintenance(vehicle_data)
    state["risk_score"] = ml_result["risk_score"]
    state["risk_label"] = ml_result["risk_label"]
    state["contributing_factors"] = ml_result["contributing_factors"]

    query = f"""
    Vehicle maintenance guidelines for:
    - Risk level: {state["risk_label"]}
    - Engine temperature: {vehicle_data.get("engine_temp", "N/A")}°C
    - Oil quality: {vehicle_data.get("oil_quality", "N/A")}
    - Battery voltage: {vehicle_data.get("battery_voltage", "N/A")}V
    - Brake condition: {vehicle_data.get("brake_condition", "N/A")}
    """
    state["retrieved_guidelines"] = retrieve_guidelines(query)

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    system_prompt = """You are a fleet maintenance advisor AI.
Your job is to generate a structured vehicle health report.
Be concise, factual, and safety-focused. Never invent sensor readings.
Always base recommendations strictly on the data provided."""
    user_prompt = f"""
Generate a structured maintenance report based on the following:

VEHICLE DATA:
{state["vehicle_data"]}

ML PREDICTION:
- Risk Label: {state["risk_label"]}
- Risk Score: {state["risk_score"]:.2%}
- Top Contributing Factors: {state["contributing_factors"]}

RETRIEVED MAINTENANCE GUIDELINES:
{state["retrieved_guidelines"]}

Respond in this exact format:

HEALTH SUMMARY:
[2-3 sentences about the vehicle's current health status and key risk indicators]

ACTION PLAN:
[3-5 bullet points with specific recommended actions and timelines]

DISCLAIMER:
[1 sentence operational safety notice]
"""
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    full_response = response.content
    state["health_summary"] = _extract_section(full_response, "HEALTH SUMMARY")
    state["action_plan"] = _extract_section(full_response, "ACTION PLAN")
    state["disclaimer"] = _extract_section(full_response, "DISCLAIMER")
    return state
