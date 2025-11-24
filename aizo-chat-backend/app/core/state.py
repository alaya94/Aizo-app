# app/core/state.py
from typing import List, TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    is_diagnosis_complete: bool
    # ✅ NEW FIELD: Stores the specific domain (e.g., "AI Adoption", "Cloud Security")
    active_strategy_topic: str


# --- ROUTER SCHEMA ---
class RouteQuery(BaseModel):
    """Route the user's query to the most relevant datasource."""
    step: Literal["files", "web_search", "general_chat", "generate_report", "strategy_planning"] = Field(
        ...,
        description="Given a user question, choose the best route.",
    )
    # ✅ NEW FIELD: The Chameleon Sensor
    strategy_topic: Optional[str] = Field(
        None, 
        description="If step is 'strategy_planning', extract the specific technical domain (e.g., 'Cloud Migration', 'ERP Upgrade', 'Cybersecurity')."
    )