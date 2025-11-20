import os
import uvicorn
import shutil
from typing import List, TypedDict, Annotated
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain / LangGraph
from redis.asyncio import Redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredImageLoader
import base64
from fastapi.responses import StreamingResponse
load_dotenv(override=True)

# --- CONFIG ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
UPLOAD_DIR = "./uploads"
DB_DIR = "./db"
PORT = 8001

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

redis_client = None 
graph = None        
vector_stores = {}
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}
# --- TOOLS ---

def get_vector_store(session_id: str):
    path = os.path.join(DB_DIR, session_id)
    if session_id not in vector_stores:
        vector_stores[session_id] = Chroma(
            persist_directory=path,
            embedding_function=OpenAIEmbeddings(),
            collection_name=f"collection_{session_id}"
        )
    return vector_stores[session_id]


async def describe_image(file_path: str) -> str:
    """
    Uses GPT-4o-mini to look at the image and describe it textually
    so we can store it in the vector DB.
    """
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # We use a direct LLM call here, distinct from the agent
    vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this image. Extract all visible text verbatim. If there are charts or graphs, describe the data trends in detail. Output ONLY the description."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
        ]
    )
    
    res = await vision_llm.ainvoke([msg])
    return res.content

@tool
def lookup_documents(query: str, config: RunnableConfig) -> str:
    """Search uploaded documents (PDFs, Word, Text) AND image descriptions.
    ALWAYS use this tool if the user asks about "this image", "the screenshot", "the graph", or "the document"."""
    print(f"\n🔎 [DEBUG] Tool 'lookup_documents' called with query: '{query}'")
    
    session_id = config.get("configurable", {}).get("thread_id")
    if not session_id: 
        print("❌ [DEBUG] No session ID found in config.")
        return "No session ID."
        
    try:
        store = get_vector_store(session_id)
        # Debug: Check how many docs are in store
        count = store._collection.count()
        print(f"📊 [DEBUG] Vector Store for '{session_id}' has {count} chunks.")
        
        docs = store.similarity_search(query, k=3)
        
        if not docs:
            print("⚠️ [DEBUG] Search returned NO results.")
            return "No relevant information found in the documents."
            
        print(f"✅ [DEBUG] Found {len(docs)} relevant chunks.")
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"❌ [DEBUG] Tool Error: {e}")
        return f"Error: {e}"

@tool
async def remember_fact(fact: str, config: RunnableConfig) -> str:
    """Store user details (name, preferences) to memory."""
    global redis_client
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id or not redis_client: return "Memory unavailable."
    await redis_client.rpush(f"memory:{thread_id}", fact)
    return f"Stored: {fact}"

# --- WORKFLOW ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

async def chatbot(state: AgentState, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    
    memories = []
    if thread_id and redis_client:
        memories = await redis_client.lrange(f"memory:{thread_id}", 0, -1)
    
    memory_text = "\n".join([f"- {m}" for m in memories]) if memories else "No memories yet."
    
    # STRONGER SYSTEM PROMPT
    system_prompt = (
        "You are a helpful AI assistant with access to a file database.\n"
        "IMPORTANT: The user may upload 'images' or 'screenshots'. These have been converted to text descriptions "
        "and stored in your document database.\n"
        "RULE: If the user asks about 'this image', 'the graph', 'the screenshot', or 'what did I upload', "
        "you MUST use the 'lookup_documents' tool to find the description. Do NOT say you cannot see images.\n\n"
        f"LONG-TERM MEMORY:\n{memory_text}"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [lookup_documents, remember_fact]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([lookup_documents, remember_fact]))
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# --- APP LIFESPAN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, redis_client
    try:
        redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
        async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
            await checkpointer.setup()
            graph = workflow.compile(checkpointer=checkpointer)
            print(f"✅ Backend running on port {PORT}")
            yield
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if redis_client: await redis_client.aclose()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not graph: raise HTTPException(503, "Graph loading...")
    
    print(f"\n💬 [DEBUG] Streaming Chat: {request.message} (Thread: {request.thread_id})")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    async def event_generator():
        try:
            # astream_events (v2) allows us to see everything happening inside the Agent
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                
                # We filter for "on_chat_model_stream" to get tokens from GPT-4
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    
                    # Only yield if there is actual text content (avoids empty chunks from Tool Calls)
                    if chunk.content:
                        yield chunk.content
                        
        except Exception as e:
            print(f"❌ [DEBUG] Stream Error: {e}")
            yield f"Error: {str(e)}"

    # Return the generator as a StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/upload/{thread_id}")
async def upload_file(thread_id: str, file: UploadFile = File(...)):
    print(f"\n📂 [DEBUG] Processing upload for Thread: {thread_id}")
    
    # 1. VALIDATE FILE EXTENSION
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # 2. SAVE TEMP FILE & VALIDATE SIZE
    file_path = os.path.join(UPLOAD_DIR, f"{thread_id}_{file.filename}")
    
    # Read file in chunks to avoid memory overload and check size
    file_size = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024): # Read 1MB chunks
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                os.remove(file_path)
                raise HTTPException(400, f"File too large. Limit is {MAX_FILE_SIZE_MB}MB.")
            buffer.write(chunk)

    docs = []
    try:
        print(f"   [DEBUG] Loading {ext} file...")
        
        # 3. ROUTE TO CORRECT LOADER
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            
        elif ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
            
        elif ext in [".png", ".jpg", ".jpeg"]:
            # Special handling for images: Ask GPT-4o to describe them
            print("   [DEBUG] Image detected. Analyzing with Vision Model...")
            description = await describe_image(file_path)
            # Create a standard LangChain Document
            docs = [Document(page_content=description, metadata={"source": file.filename})]
        
        # 4. HANDLE SCANNED PDFS (Empty Text)
        # If PyPDFLoader returns docs but they are empty strings, it's a scanned PDF.
        if ext == ".pdf" and docs and len(docs[0].page_content.strip()) < 10:
            print("⚠️ [DEBUG] PDF appears to be scanned (no text). You might need OCR.")
            # Note: To handle scanned PDFs, you'd need 'pdf2image' library 
            # to convert pages to images, then send to describe_image() function above.
            # For now, we just warn.
            return {"status": "warning", "detail": "PDF contains no text (Scanned?)."}

        if not docs:
            return {"status": "error", "detail": "No content extracted."}

        # 5. SPLIT & STORE
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        store = get_vector_store(thread_id)
        store.add_documents(splits)
        print(f"✅ [DEBUG] Added {len(splits)} chunks to store.")

    except Exception as e:
        print(f"❌ [DEBUG] Upload Error: {e}")
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return {"status": "success", "chunks": len(splits)}


@app.delete("/delete_thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    print(f"\n🗑️ [DEBUG] Request to delete thread: {thread_id}")
    
    # 1. Delete Vector Store (ChromaDB)
    try:
        path = os.path.join(DB_DIR, thread_id)
        if os.path.exists(path):
            shutil.rmtree(path) # Nuke the folder
            if thread_id in vector_stores:
                del vector_stores[thread_id]
            print("   ✅ Vector Store deleted.")
        else:
            print("   ⚠️ Vector Store not found (already deleted?)")
    except Exception as e:
        print(f"   ❌ Error deleting Vector Store: {e}")

    # 2. Delete Long-Term Memory (Redis)
    try:
        if redis_client:
            # Delete our custom memory list
            await redis_client.delete(f"memory:{thread_id}")
            # Optional: Delete LangGraph checkpoints (if you want to wipe conversation history too)
            # This depends on how LangGraph stores keys, but usually specific cleanup is complex. 
            # Deleting the memory list is the most important part.
            print("   ✅ Redis Memory deleted.")
    except Exception as e:
        print(f"   ❌ Error deleting Redis Memory: {e}")

    return {"status": "deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)