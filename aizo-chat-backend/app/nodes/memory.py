# app/nodes/memory.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from app.config import MAIN_MODEL, MAX_PROFILE_TOKENS, SUMMARY_THRESHOLD
from app.core.state import AgentState
from app.core.prompts import (
    PROFILE_INSTRUCTIONS, 
    COMPRESSION_PROMPT,
    SUMMARY_EXTEND_PROMPT,
    SUMMARY_CREATE_PROMPT
)
from app.utils.tokens import count_tokens
from app.dependencies import get_redis


async def update_profile_node(state: AgentState, config: RunnableConfig) -> dict:
    """Updates the persistent user profile with self-compression logic."""
    
    user_id = "admin"
    redis = get_redis()
    
    if not redis:
        return {}

    # Get existing profile
    current_profile = await redis.get(f"user_profile:{user_id}") or "No existing history."
    
    # Get recent messages
    recent_messages = state["messages"][-2:]
    conversation_text = "\n".join([
        f"{m.type}: {m.content}" 
        for m in recent_messages 
        if hasattr(m, 'content') and m.content
    ])

    # Generate updated profile
    model = ChatOpenAI(model=MAIN_MODEL, temperature=0)
    prompt = PROFILE_INSTRUCTIONS.format(
        history=current_profile, 
        new_lines=conversation_text
    )
    
    response = await model.ainvoke([HumanMessage(content=prompt)])
    new_profile = response.content

    # Token limit enforcement
    token_count = count_tokens(new_profile)

    if token_count > MAX_PROFILE_TOKENS:
        print(f"⚠️ [MEMORY] Profile too large ({token_count} tokens). Compressing...")
        
        compression_prompt = COMPRESSION_PROMPT.format(
            token_count=token_count,
            profile=new_profile
        )
        
        compressed_response = await model.ainvoke([HumanMessage(content=compression_prompt)])
        new_profile = compressed_response.content
        final_count = count_tokens(new_profile)
        print(f"✅ [MEMORY] Compressed to {final_count} tokens.")

    # Save to Redis
    await redis.set(f"user_profile:{user_id}", new_profile)
    
    return {}


async def summarize_conversation_node(state: AgentState, config: RunnableConfig) -> dict:
    """Summarizes conversation while preserving tool call/response integrity."""
    
    messages = state["messages"]
    
    # Only summarize if history is getting long
    if len(messages) <= SUMMARY_THRESHOLD:
        return {}

    summary = state.get("summary", "")
    
    # Find safe truncation point - never break tool call/response pairs
    safe_delete_index = 0
    i = 0
    
    while i < len(messages) - 4:  # Keep at least last 4 messages
        msg = messages[i]
        
        # Skip tool_calls message AND its response together
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            i += 2
            continue
        
        # Skip tool messages
        if msg.type == "tool":
            i += 1
            continue
            
        safe_delete_index = i + 1
        i += 1
    
    messages_to_summarize = messages[:safe_delete_index]
    
    if len(messages_to_summarize) < 2:
        return {}

    # Build summary prompt
    if summary:
        sys_msg = SUMMARY_EXTEND_PROMPT.format(summary=summary)
    else:
        sys_msg = SUMMARY_CREATE_PROMPT

    model = ChatOpenAI(model=MAIN_MODEL, temperature=0)
    
    # Filter out tool messages for summarization
    summarizable = [
        m for m in messages_to_summarize 
        if m.type in ["human", "ai"] and not (hasattr(m, 'tool_calls') and m.tool_calls)
    ]
    
    if not summarizable:
        return {}
        
    response = await model.ainvoke(summarizable + [HumanMessage(content=sys_msg)])
    new_summary = response.content

    # Create delete messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    print(f"✂️ [DEBUG] Summarized context. Removing {len(delete_messages)} old messages.")
    
    return {
        "summary": new_summary, 
        "messages": delete_messages
    }