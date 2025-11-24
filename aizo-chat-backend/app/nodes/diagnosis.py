
from langchain_core.runnables import RunnableConfig
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.prompts import DIAGNOSIS_SYSTEM_PROMPT
async def diagnosis_node(state: AgentState, config: RunnableConfig):
    # 1. READ THE TOPIC
    topic = state.get("active_strategy_topic", "Digital Transformation")
    print(f"🩺 [DEBUG] Diagnosing for specific topic: {topic}")
    
    # Safety break
    if state.get("is_diagnosis_complete"):
        return {}

    messages = state["messages"]
    
    # --- CRITICAL FIX START ---
    # Convert the list of Message objects into a readable text history.
    # We take the last 6 messages to ensure we have context without blowing the token limit.
    recent_msgs = messages[-6:] 
    conversation_history = "\n".join([f"{m.type.upper()}: {m.content}" for m in recent_msgs])
    diagnosis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Pass the HISTORY, not just the last message
    verification = await diagnosis_llm.ainvoke(
        [
            SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"Conversation History:\n{conversation_history}")
        ],
        config=config
    )
    
    response = verification.content
    
    if "DIAGNOSIS_COMPLETE" in response:
        print("   ✅ Diagnosis Complete.")
        return {"is_diagnosis_complete": True}
    else:
        print("   ❓ Diagnosis Incomplete. Asking user...")
        return {
            "messages": [AIMessage(content=response)], 
            "is_diagnosis_complete": False
        }