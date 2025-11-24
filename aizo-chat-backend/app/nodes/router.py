# app/nodes/router.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.config import ROUTER_MODEL
from app.core.state import AgentState, RouteQuery
from app.core.prompts import ROUTER_SYSTEM_PROMPT
from langgraph.graph import StateGraph, END

async def router_node(state: AgentState, config: RunnableConfig):
    """
    Analyzes the user's last message and decides the best tool strategy.
    """
    print("🚦 [DEBUG] Router determining intent...")
    messages = state["messages"]
    last_msg = messages[-1].content
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # 1. Update the System Prompt to teach it about Reports
    system = (
        "You are a router. Classify the user's intent.\n"
        "- 'generate_report' for pdfs/downloads.\n"
        "- 'strategy_planning' for advice, strategy, implementation plans, or 'how to'.\n"
        "- 'files' for uploaded documents.\n"
        "- 'web_search' for trends/news.\n"
        "- 'general_chat' for hi/definitions."
    )
    
    route = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=last_msg)
    ])
    
    step = route.step
    topic = route.strategy_topic or "Digital Transformation"
    
    print(f"   → Routing to: {step.upper()} (Topic: {topic})")
    
    directive = ""
    
    # 2. Add the Logic for the New Intent
    if step == "strategy_planning":
        directive = (
            f"INTENT DETECTED: {topic.upper()} STRATEGY. "
            "Use the diagnosis info to build a tailored plan."
        )
    elif step == "generate_report":
        directive = (
            "INTENT DETECTED: REPORT GENERATION. "
            "Step 1: If you need data, use 'research_digital_trends' or 'lookup_documents' first. "
            "Step 2: You MUST use the 'generate_report' tool to create the PDF."
        )
        # We append the directive so the context is preserved
        return {
            "messages": [SystemMessage(content=directive)],
            # OPTIONAL: If you want to force diagnosis specifically for reports
            # "force_diagnosis": True 
        }
    elif step == "general_chat":
        # LOGIC CHANGE: If the user asks a complex strategy question, we send to diagnosis
        # Ideally, your prompt above should have a category for "strategy_consulting"
        # For now, let's assume 'general_chat' might contain strategy needs.
        
        # If you implemented the 'diagnosis_node' wiring, you likely want to 
        # return a specific signal here if the query is complex.
        directive = "INTENT DETECTED: GENERAL CONVERSATION. Do NOT use any tools. Just reply."

    elif step == "files":
        directive = "INTENT DETECTED: FILE QUERY. You MUST use the 'lookup_documents' tool."

    elif step == "web_search":
        directive = "INTENT DETECTED: WEB RESEARCH. You MUST use the 'research_digital_trends' tool."
        
    return {
        "messages": [SystemMessage(content=directive)],
        "active_strategy_topic": topic # <--- Saving to State
    }


def route_based_on_intent(state: AgentState):
    """Decides if we need Diagnosis or go straight to Chatbot."""
    messages = state["messages"]
    # The Router node appends a SystemMessage with the directive at the end
    last_msg = messages[-1].content
    
    # Logic: If Report or complex Strategy, go to Diagnosis first
    # (Ensure your router_node actually puts these strings in the directive)
    if "REPORT GENERATION" in last_msg or "STRATEGY" in last_msg:
        return "diagnosis"
    
    # Default: Go to Chatbot (for Search, Files, or Chat)
    return "chatbot"

def check_diagnosis_status(state: AgentState):
    """After Diagnosis, do we have enough info to proceed?"""
    if state.get("is_diagnosis_complete"):
        return "chatbot" # Success: Go to Agent to solve
    return END
def chatbot_logic(state: AgentState):
    """Decides if Chatbot needs Tools or is finished."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "update_profile"