from typing import TypedDict


class AgentState(TypedDict):
    # Input vehicle data
    vehicle_data: dict

    # ML prediction results
    risk_score: float
    risk_label: str
    contributing_factors: dict

    # RAG results
    retrieved_guidelines: str

    # Final LLM output
    health_summary: str
    action_plan: str
    disclaimer: str
