import os
import uvicorn
import shutil
from typing import List, TypedDict, Annotated, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from redis.asyncio import Redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredImageLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import base64
import tiktoken

from langchain_community.tools import TavilySearchResults

load_dotenv(override=True)

# --- CONFIG ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
UPLOAD_DIR = "./uploads"
DB_DIR = "./db"
PORT = 8001
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

redis_client = None 
graph = None        
vector_stores = {}

# --- PROMPTS ---

PROFILE_INSTRUCTIONS = """
You are a memory manager for Aizo, a Digital Transformation expert. 
Your job is to update the professional profile of the user.

Current Profile:
{history}

New conversation lines:
{new_lines}

INSTRUCTIONS:
- Extract ONLY professional details: Job Title, Industry, Company Goals, Technical Stack, Challenges, and Digital Strategy preferences.
- IGNORE personal trivia (e.g., pets, food, hobbies) unless it relates to their business persona.
- Merge new facts into the Current Profile.
- Keep the output as a concise bulleted list.
"""

# --- TOOLS (Unchanged) ---
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
def get_vector_store(session_id: str):
    path = os.path.join(DB_DIR, session_id)
    if session_id not in vector_stores:
        vector_stores[session_id] = Chroma(
            persist_directory=path,
            embedding_function=OpenAIEmbeddings(),
            collection_name=f"collection_{session_id}"
        )
    return vector_stores[session_id]

@tool
def lookup_documents(query: str, config: RunnableConfig) -> str:
    """Search uploaded documents (PDFs, Word, Text) AND image descriptions."""
    print(f"\n🔎 [DEBUG] Tool 'lookup_documents' called: '{query}'")
    session_id = config.get("configurable", {}).get("thread_id")
    if not session_id: return "No session ID."
    try:
        store = get_vector_store(session_id)
        docs = store.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs]) if docs else "No info found."
    except Exception as e:
        return f"Error: {e}"
@tool
def research_digital_trends(query: str) -> str:
    """
    Use this tool to find REAL-TIME statistics, trends, and news about Digital Transformation.
    Useful for questions like 'What are the AI trends in 2025?' or 'Latest cloud adoption stats'.
    Sources are restricted to trusted tech & consulting firms (Gartner, McKinsey, TechCrunch, etc.).
    """
    print(f"\n🌐 [DEBUG] Tool 'research_digital_trends' called: '{query}'")
    
    try:
        # We limit the search to high-quality domains to ensure "Expert" quality
        search_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_domains=[
                "gartner.com", 
                "mckinsey.com", 
                "forrester.com", 
                "bcg.com", 
                "hbr.org", 
                "techcrunch.com", 
                "venturebeat.com",
                "aws.amazon.com",
                "azure.microsoft.com"
            ]
        )
        # Execute the search
        results = search_tool.invoke({"query": query})
        
        # Format results for the LLM
        output = []
        for res in results:
            output.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
            
        return "\n\n".join(output)
        
    except Exception as e:
        return f"Search Error: {e}"
# --- STATE DEFINITION ---

class AgentState(TypedDict):
    # add_messages handles the append logic
    messages: Annotated[List[BaseMessage], add_messages]
    # summary holds the Short-Term context summary
    summary: str

# --- NODES ---

# 1. CHATBOT NODE (Consumes Memories)
async def chatbot(state: AgentState, config: RunnableConfig):
    # Helper to get User ID (for Profile) and Thread ID (for Vector Store)
    conf = config.get("configurable", {})
    thread_id = conf.get("thread_id")
    user_id = "admin" # In a real app, pass this in config. For now, we default to admin.

    # A. FETCH LONG-TERM PROFILE (User Scoped)
    user_profile = ""
    if redis_client:
        # Note: Key is based on USER_ID, not THREAD_ID. This is shared memory!
        user_profile = await redis_client.get(f"user_profile:{user_id}")
        if not user_profile:
            user_profile = "No profile yet."
    
    # B. FETCH SHORT-TERM SUMMARY (Thread Scoped)
    short_term_context = state.get("summary", "")

    # C. CONSTRUCT SYSTEM PROMPT
# ... inside chatbot function ...

    # C. CONSTRUCT SYSTEM PROMPT (THE AIZO PERSONALITY)
    system_prompt = (
        "### IDENTITY ###\n"
        "Your name is **Aizo**. You are a Senior Consultant and Expert in **Digital Transformation**.\n"
        "You are professional, insightful, and strategic. You speak with the authority of a CIO or Tech Strategist.\n\n"

        "### SCOPE OF EXPERTISE ###\n"
        "You are programmed to ONLY answer questions related to:\n"
        "- Digital Transformation & Strategy\n"
        "- AI Adoption & Machine Learning\n"
        "- Cloud Computing & Infrastructure\n"
        "- Process Automation & Legacy Modernization\n"
        "- Organizational Change Management in Tech\n"
        "- Software Engineering & Architecture\n\n"

        "### GUARDRAILS & REFUSAL ###\n"
        "If the user asks about topics outside this scope (e.g., cooking, sports, politics, general trivia, weather):\n"
        "1. Politely refuse.\n"
        "2. Say: 'I am Aizo, a specialist in Digital Transformation. I cannot assist with [Topic], but I can help you digitize your business processes or discuss AI strategy.'\n"
        "3. Do NOT generate content for off-topic requests.\n\n"

        "### CONTEXT & TOOLS ###\n"
        "- Use 'lookup_documents' if the user asks about uploaded files.\n"
        "- Use the User Profile below to personalize your strategic advice.\n\n"

        "--- LONG TERM USER PROFILE ---\n"
        f"{user_profile}\n"
        "------------------------------\n"
        "--- CURRENT CONVERSATION SUMMARY ---\n"
        f"{short_term_context}\n"
        "------------------------------------"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([lookup_documents,research_digital_trends])
    
    # We only pass the System Prompt + The (already trimmed) messages in state
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


# 2. LONG-TERM MEMORY UPDATER (The Profile Builder)
# --- HELPER: TOKEN COUNTER ---
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# --- ROBUST MEMORY NODE ---
async def update_profile_node(state: AgentState, config: RunnableConfig):
    """Updates the persistent user profile, with self-compression logic."""
    user_id = "admin" 
    if not redis_client: return {}

    # 1. Get existing data
    current_profile = await redis_client.get(f"user_profile:{user_id}") or "No existing history."
    recent_messages = state["messages"][-2:] 
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])

    # 2. Generate Updated Profile
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # We add a constraint to the prompt to encourage brevity
    prompt = PROFILE_INSTRUCTIONS.format(history=current_profile, new_lines=conversation_text)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    new_profile = response.content

    # 3. THE SAFETY CHECK (Token Limit Enforcement)
    # Limit: 1000 tokens (approx 750 words). 
    token_count = count_tokens(new_profile)
    MAX_PROFILE_TOKENS = 1000

    if token_count > MAX_PROFILE_TOKENS:
        print(f"⚠️ [MEMORY] Profile too large ({token_count} tokens). Compressing...")
        
        compression_prompt = (
            f"The following user profile is too long ({token_count} tokens). \n"
            "Rewrite it to be strictly under 700 tokens.\n"
            "Maintain ALL key facts (names, preferences, technical details), but remove fluff, "
            "redundant sentences, and poetic language. Use concise bullet points.\n\n"
            f"CURRENT PROFILE:\n{new_profile}"
        )
        
        compressed_response = await model.ainvoke([HumanMessage(content=compression_prompt)])
        new_profile = compressed_response.content
        final_count = count_tokens(new_profile)
        print(f"✅ [MEMORY] Compressed to {final_count} tokens.")

    # 4. Save
    await redis_client.set(f"user_profile:{user_id}", new_profile)
    return {}


# 3. SHORT-TERM MEMORY SUMMARIZER (The Context Cleaner)
async def summarize_conversation_node(state: AgentState, config: RunnableConfig):
    """Summarizes conversation and keeps only the last 2 exchanges."""
    
    messages = state["messages"]
    
    # Only summarize if history is getting long (e.g., > 4 messages)
    if len(messages) <= 4:
        return {}

    summary = state.get("summary", "")
    
    # Create prompt to summarize
# Create prompt to summarize
    if summary:
        sys_msg = (
            f"Current Summary: {summary}\n\n"
            "Extend this summary by merging in the new lines below.\n"
            "STRICT CONSTRAINT: Keep the total summary under 400 words.\n"
            "Focus only on the unresolved questions and current topic flow."
        )
    else:
        sys_msg = "Create a concise summary of the conversation above (max 400 words)."

    # We summarize everything EXCEPT the last 2 messages (which we want to keep raw)
    messages_to_summarize = messages[:-2]
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await model.ainvoke(messages_to_summarize + [HumanMessage(content=sys_msg)])
    new_summary = response.content

    # DELETE old messages from the state
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    print(f"✂️ [DEBUG] Summarized context. Removing {len(delete_messages)} old messages.")
    
    return {
        "summary": new_summary, 
        "messages": delete_messages # This creates the deletion effect in LangGraph
    }


# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([lookup_documents, research_digital_trends]))
workflow.add_node("update_profile", update_profile_node)
workflow.add_node("summarize_conversation", summarize_conversation_node)


# Entry
workflow.set_entry_point("chatbot")

# Logic
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "update_profile" # If no tool needed, go to memory maintenance

workflow.add_conditional_edges("chatbot", should_continue)
workflow.add_edge("tools", "chatbot")

# Memory Maintenance Chain
# After answering, update Profile -> Then Clean Context -> End
workflow.add_edge("update_profile", "summarize_conversation")
workflow.add_edge("summarize_conversation", END)

# --- APP LIFESPAN (Standard) ---
# ... [Keep your existing imports/lifespan/endpoint code below exactly as is] ...
# ... Just make sure to include the helper/upload functions from previous steps ...

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    thread_id: str

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not graph: raise HTTPException(503, "Graph loading...")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    async def event_generator():
        try:
            # We stream events from the graph
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                
                # 1. Check if this event comes from the 'chatbot' node
                #    (We do NOT want to stream tokens from 'update_profile' or 'summarize')
                kind = event["event"]
                tags = event.get("tags", [])
                metadata = event.get("metadata", {})
                
                # CRITICAL FIX: Filter by node name
                if kind == "on_chat_model_stream" and metadata.get("langgraph_node") == "chatbot":
                    
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
                        
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"Error: {str(e)}"

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