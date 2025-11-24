# app/nodes/chatbot.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import MAIN_MODEL
from app.core.state import AgentState
from app.core.prompts import SYSTEM_PROMPT_TEMPLATE
from app.nodes.tools import ALL_TOOLS
from app.dependencies import get_redis


async def chatbot(state: AgentState, config: RunnableConfig) -> dict:
    """Main chatbot node that processes messages and decides on tool usage."""
    
    conf = config.get("configurable", {})
    thread_id = conf.get("thread_id")
    user_id = "admin"

    # Fetch long-term profile from Redis
    user_profile = "No profile yet."
    redis = get_redis()
    if redis:
        profile = await redis.get(f"user_profile:{user_id}")
        if profile:
            user_profile = profile
    
    # Get short-term context from state
    short_term_context = state.get("summary", "")

    # Build system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_profile=user_profile,
        short_term_context=short_term_context
    )
    
    # Initialize LLM with tools
    llm = ChatOpenAI(model=MAIN_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    
    # Prepare messages
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Get response
    response = await llm_with_tools.ainvoke(messages)
    
    if response.tool_calls:
        print(f"\n🛠️ [DEBUG] AI is calling tools: {response.tool_calls}")
    
    return {"messages": [response]}