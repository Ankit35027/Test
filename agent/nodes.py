import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.state import AgentState
from ml.predict import predict_maintenance
from rag.retriever import retrieve_guidelines


load_dotenv()


def ml_prediction_node(state: AgentState) -> AgentState:
    """Node 1: Run the ML model to get the maintenance risk prediction."""
    vehicle_data = state["vehicle_data"]
    result = predict_maintenance(vehicle_data)

    state["risk_score"] = result["risk_score"]
    state["risk_label"] = result["risk_label"]
    state["contributing_factors"] = result["contributing_factors"]
    return state


def rag_retrieval_node(state: AgentState) -> AgentState:
    """Node 2: Retrieve relevant maintenance guidelines via RAG."""
    vehicle_data = state["vehicle_data"]

    query = f"""
    Vehicle maintenance guidelines for:
    - Risk level: {state["risk_label"]}
    - Engine temperature: {vehicle_data.get("engine_temp", vehicle_data.get("Engine_Temperature", "N/A"))}°C
    - Oil quality: {vehicle_data.get("oil_quality", vehicle_data.get("Oil_Quality", "N/A"))}
    - Battery voltage: {vehicle_data.get("battery_voltage", vehicle_data.get("Battery_Voltage", "N/A"))}V
    - Brake condition: {vehicle_data.get("brake_condition", vehicle_data.get("Brake_Condition", "N/A"))}
    """

    state["retrieved_guidelines"] = retrieve_guidelines(query)
    return state


def llm_summary_node(state: AgentState) -> AgentState:
    """Node 3: Use an LLM to synthesize a structured fleet report."""
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
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

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    full_response = response.content

    def extract_section(text: str, section_name: str) -> str:
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

    state["health_summary"] = extract_section(full_response, "HEALTH SUMMARY")
    state["action_plan"] = extract_section(full_response, "ACTION PLAN")
    state["disclaimer"] = extract_section(full_response, "DISCLAIMER")
    return state
