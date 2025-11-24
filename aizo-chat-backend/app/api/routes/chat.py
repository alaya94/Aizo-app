# app/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.dependencies import get_graph

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages with streaming response."""
    
    graph = get_graph()
    if not graph:
        raise HTTPException(503, "Graph not initialized")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    async def event_generator():
        try:
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                metadata = event.get("metadata", {})
                
                # Only stream tokens from the chatbot node
                if kind == "on_chat_model_stream" and metadata.get("langgraph_node") == "chatbot":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
                        
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")