# app/core/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.state import AgentState
from app.nodes.router import router_node
from app.nodes.chatbot import chatbot
from app.nodes.memory import update_profile_node, summarize_conversation_node
from app.nodes.tools import ALL_TOOLS
from app.nodes.diagnosis import diagnosis_node
from app.nodes.router import route_based_on_intent, check_diagnosis_status, chatbot_logic


def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or finalize."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "update_profile"


def build_graph() -> StateGraph:
    """Build and return the workflow graph (uncompiled)."""
    
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("diagnosis", diagnosis_node)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_node("update_profile", update_profile_node)
    workflow.add_node("summarize_conversation", summarize_conversation_node)

    # Set entry point
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
    "router",
    route_based_on_intent,
    {
        "diagnosis": "diagnosis",
        "chatbot": "chatbot"
    }
)
    workflow.add_conditional_edges(
    "diagnosis",
    check_diagnosis_status,
    {
        "chatbot": "chatbot", # Info gathered -> Solve
        END: END              # Need user input -> Stop
    }
)
    workflow.add_conditional_edges(
    "chatbot",
    chatbot_logic,
    {
        "tools": "tools",
        "update_profile": "update_profile"
    }
)
    # Define edges
    workflow.add_edge("tools", "chatbot")

    # 7. Memory Chain (Linear)
    workflow.add_edge("update_profile", "summarize_conversation")
    workflow.add_edge("summarize_conversation", END)

    return workflow