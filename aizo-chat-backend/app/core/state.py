# app/core/state.py
from typing import List, TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """Main state for the agent graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str


# --- ROUTER SCHEMA ---
class RouteQuery(BaseModel):
    """Route the user's query to the most relevant datasource."""
    # ✅ Add "generate_report" to this list
    step: Literal["files", "web_search", "general_chat", "generate_report"] = Field(
        ...,
        description="Given a user question, choose to route it to web search, uploaded files, general chat, or report generation.",
    )