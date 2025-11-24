# app/nodes/router.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.config import ROUTER_MODEL
from app.core.state import AgentState, RouteQuery
from app.core.prompts import ROUTER_SYSTEM_PROMPT


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
        "You are a router. Your job is to classify the user's intent.\n"
        "- If they want to 'generate a report', 'create a pdf', or 'download file' -> 'generate_report'.\n"
        "- If they ask about 'this file', 'uploaded document', or specific project details -> 'files'.\n"
        "- If they ask about 'trends', 'news', 'stats', '2025', 'market' -> 'web_search'.\n"
        "- If they say 'hi', 'who are you', or ask general theoretical questions -> 'general_chat'."
    )
    
    route = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=last_msg)
    ])
    
    step = route.step
    print(f"   → Routing to: {step.upper()}")
    
    directive = ""
    
    # 2. Add the Logic for the New Intent
    if step == "generate_report":
        # Crucial: Give permission to Search FIRST, then Report
        directive = (
            "INTENT DETECTED: REPORT GENERATION. "
            "Step 1: If you need data, use 'research_digital_trends' or 'lookup_documents' first. "
            "Step 2: You MUST use the 'generate_report' tool to create the PDF."
        )
    elif step == "files":
        directive = "INTENT DETECTED: FILE QUERY. You MUST use the 'lookup_documents' tool."
    elif step == "web_search":
        directive = "INTENT DETECTED: WEB RESEARCH. You MUST use the 'research_digital_trends' tool."
    else:
        directive = "INTENT DETECTED: GENERAL CONVERSATION. Do NOT use any tools. Just reply."
        
    return {"messages": [SystemMessage(content=directive)]}