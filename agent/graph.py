from langgraph.graph import END, StateGraph

from agent.nodes import llm_summary_node, ml_prediction_node, rag_retrieval_node
from agent.state import AgentState


def build_fleet_agent():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("ml_prediction", ml_prediction_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("llm_summary", llm_summary_node)

    graph.set_entry_point("ml_prediction")
    graph.add_edge("ml_prediction", "rag_retrieval")
    graph.add_edge("rag_retrieval", "llm_summary")
    graph.add_edge("llm_summary", END)

    return graph.compile()


fleet_agent = build_fleet_agent()
