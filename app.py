import os
import uvicorn
import shutil
from typing import List, TypedDict, Annotated, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
from typing import Literal
# from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch
# --- ADD THESE IMPORTS ---
from fastapi.staticfiles import StaticFiles  # To serve the PDF files
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import time
from langsmith import traceable
# --- UPDATE CONFIGURATION ---
# Add a directory for generated reports
DOWNLOAD_DIR = "./downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ... (Keep existing REDIS_URL, etc.) ...

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
# --- ROUTER SCHEMA ---
# --- ROUTER SCHEMA ---
class RouteQuery(BaseModel):
    """Route the user's query to the most relevant datasource."""
    # ✅ Add "generate_report" to this list
    step: Literal["files", "web_search", "general_chat", "generate_report"] = Field(
        ...,
        description="Given a user question, choose to route it to web search, uploaded files, general chat, or report generation.",
    )
# --- STATE DEFINITION ---

class AgentState(TypedDict):
    # add_messages handles the append logic
    messages: Annotated[List[BaseMessage], add_messages]
    # summary holds the Short-Term context summary
    summary: str
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
@traceable(name="Intent Router")
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
    """
    print(f"\n🌐 [DEBUG] Tool 'research_digital_trends' called: '{query}'")
    
    try:
        search_tool = TavilySearch(
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

        print("🔍 [DEBUG] Executing Tavily search request...")
        response = search_tool.invoke({"query": query})

        print("📦 [DEBUG] Raw Tavily response:")
        print(response)

        # --- FIX: Tavily returns a dict, and results are inside response["results"] ---
        tavily_results = response.get("results", [])

        if not tavily_results:
            return "No results found."

        formatted_outputs = []

        for item in tavily_results:
            url = item.get("url", "No URL")
            content = item.get("content", "No Content")
            formatted_outputs.append(f"Source: {url}\nContent: {content}")

        print("✅ [DEBUG] Formatted results generated.")

        return "\n\n".join(formatted_outputs)

    except Exception as e:
        print("❌ [DEBUG] ERROR OCCURRED:", e)
        return f"Search Error: {e}"


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import re
@tool
def generate_report(topic: str, content: str) -> str:
    """
    Generates a professional PDF report. 
    Parses Markdown (headers #, ##, ###, bold **, lists -) into a clean PDF layout.
    """
    print(f"📝 [DEBUG] Generating Formatted PDF: {topic}")
    
    try:
        filename = f"Report_{int(time.time())}.pdf"
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        
        # 1. Setup Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, spaceAfter=12))
        
        story = []
        
        # 2. Add Main Title
        story.append(Paragraph(topic, styles['Title']))
        story.append(Spacer(1, 24))
        
        # 3. PARSER: Convert Markdown to ReportLab Flowables
        lines = content.split('\n')
        current_list = []
        
        def flush_list(buffer, story_obj):
            if buffer:
                story_obj.append(ListFlowable(buffer, bulletType='bullet', start='circle', leftIndent=20))
                story_obj.append(Spacer(1, 12))
                buffer.clear()
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # --- Pre-processing: Bold and Italics ---
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
            
            # --- A. Handle Headers (H1, H2, H3, H4) ---
            if line.startswith('# '):
                flush_list(current_list, story)     # Ensure no list is pending
                text = line.replace('# ', '')
                story.append(Paragraph(text, styles['Heading1']))
                story.append(Spacer(1, 12))
                
            elif line.startswith('## '):
                flush_list(current_list, story)
                text = line.replace('## ', '')
                story.append(Paragraph(text, styles['Heading2']))
                story.append(Spacer(1, 10))

            # --- NEW: Handle Heading 3 (###) ---
            elif line.startswith('### '):
                flush_list(current_list, story)
                text = line.replace('### ', '')
                # ReportLab usually has Heading3, but we ensure it looks distinct
                story.append(Paragraph(text, styles['Heading3']))
                story.append(Spacer(1, 8))

            # --- NEW: Handle Heading 4 (####) ---
            elif line.startswith('#### '):
                flush_list(current_list, story)
                text = line.replace('#### ', '')
                story.append(Paragraph(text, styles['Heading4']))
                story.append(Spacer(1, 6))
            
            # --- B. Handle Bullet Points (- or *) ---
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:] 
                current_list.append(ListItem(Paragraph(text, styles['Normal'])))
            
            # --- C. Handle Numbered Lists (1. ) ---
            elif re.match(r'^\d+\.', line):
                text = line.split('.', 1)[1].strip()
                current_list.append(ListItem(Paragraph(text, styles['Normal'])))

            # --- D. Normal Paragraphs ---
            else:
                flush_list(current_list, story) # Paragraph breaks the list
                story.append(Paragraph(line, styles['Justify']))
        
        # Final flush at end of document
        flush_list(current_list, story)

        # 4. Build PDF
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        doc.build(story)
        
        download_url = f"http://localhost:{PORT}/downloads/{filename}"
        return f"✅ Report generated successfully! Download here: {download_url}"

    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return f"Error generating report: {e}"
@traceable
async def chatbot(state: AgentState, config: RunnableConfig):
    conf = config.get("configurable", {})
    thread_id = conf.get("thread_id")
    user_id = "admin"

    # A. FETCH LONG-TERM PROFILE
    user_profile = ""
    if redis_client:
        user_profile = await redis_client.get(f"user_profile:{user_id}") or "No profile yet."
    
    # B. FETCH SHORT-TERM SUMMARY
    short_term_context = state.get("summary", "")

    # C. SYSTEM PROMPT
    system_prompt = (
        "### IDENTITY ###\n"
        "Your name is **Aizo**. You are a Senior Consultant in **Digital Transformation**.\n\n"
        "### TOOLS ###\n"
        "1. **Knowledge:** Use 'lookup_documents' for files.\n"
        "2. **Research:** Use 'research_digital_trends' for live web data.\n"
        "3. **Reporting:** Use 'generate_report' ONLY when the user explicitly asks to 'download', 'create a file', or 'generate a report'.\n"
        "   - When using this tool, pass a professional title and a DETAILED summary of the advice given so far.\n\n"
        "### USER PROFILE ###\n"
        f"{user_profile}\n"
        "------------------------------\n"
        f"{short_term_context}\n"
    )
    
    # --- STABILIZATION FIX ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # bind_tools(..., parallel_tool_calls=False) FORCES the AI to be careful/sequential
    llm_with_tools = llm.bind_tools(
        [lookup_documents, research_digital_trends, generate_report], # <--- Added here
        parallel_tool_calls=False 
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # --- DEBUG: SEE WHAT IS HAPPENING ---
    response = await llm_with_tools.ainvoke(messages)
    
    if response.tool_calls:
        print(f"\n🛠️ [DEBUG] AI is calling tools: {response.tool_calls}")
    
    return {"messages": [response]}

# --- HELPER: IMAGE SUMMARIZER ---
async def describe_image(file_path: str) -> str:
    """
    Uses GPT-4o-mini to look at the image and describe it textually.
    """
    print(f"   [DEBUG] Vision processing started for: {file_path}")
    
    try:
        # 1. Open file safely
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 2. Call OpenAI Vision
        vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
        
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this image. Extract all visible text verbatim. Describe charts/graphs in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        )
        
        print("   [DEBUG] Sending image to OpenAI...")
        res = await vision_llm.ainvoke([msg])
        print("   [DEBUG] Vision response received.")
        return res.content

    except Exception as e:
        print(f"❌ [VISION ERROR] Could not analyze image: {e}")
        # Fallback: Return a placeholder so the whole upload doesn't crash
        return "Error analyzing image. (Check backend logs for details)"
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
    """Summarizes conversation while PRESERVING tool call/response integrity."""
    
    messages = state["messages"]
    
    # Only summarize if history is getting long
    if len(messages) <= 6:  # Increased threshold
        return {}

    summary = state.get("summary", "")
    
    # CRITICAL FIX: Find safe truncation point
    # Never delete a tool_calls message without its corresponding ToolMessage
    safe_delete_index = 0
    i = 0
    while i < len(messages) - 4:  # Keep at least last 4 messages
        msg = messages[i]
        
        # If this is an AI message with tool_calls, skip it AND the next message (tool response)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            i += 2  # Skip both the tool_calls and the tool response
            continue
        
        # If this is a ToolMessage, it should have been handled above, but skip anyway
        if msg.type == "tool":
            i += 1
            continue
            
        # Safe to potentially delete this message
        safe_delete_index = i + 1
        i += 1
    
    # Only proceed if we have messages we can safely delete
    messages_to_summarize = messages[:safe_delete_index]
    
    if len(messages_to_summarize) < 2:
        return {}  # Nothing safe to summarize
    
    # Create summary
    if summary:
        sys_msg = (
            f"Current Summary: {summary}\n\n"
            "Extend this summary by merging in the new lines below.\n"
            "Keep under 400 words. Focus on key topics and decisions."
        )
    else:
        sys_msg = "Create a concise summary of the conversation above (max 400 words)."

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Filter out tool messages for summarization (they don't summarize well)
    summarizable = [m for m in messages_to_summarize if m.type in ["human", "ai"] and not (hasattr(m, 'tool_calls') and m.tool_calls)]
    
    if not summarizable:
        return {}
        
    response = await model.ainvoke(summarizable + [HumanMessage(content=sys_msg)])
    new_summary = response.content

    # Delete old messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    print(f"✂️ [DEBUG] Summarized context. Removing {len(delete_messages)} old messages.")
    
    return {
        "summary": new_summary, 
        "messages": delete_messages
    }

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode([lookup_documents, research_digital_trends,generate_report]))
workflow.add_node("update_profile", update_profile_node)
workflow.add_node("summarize_conversation", summarize_conversation_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "chatbot")

# FIXED: Only go to memory nodes when conversation is COMPLETE (no more tool calls)
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "update_profile"  # Only update memory when done with tools

workflow.add_conditional_edges("chatbot", should_continue)
workflow.add_edge("tools", "chatbot")  # Loop back to chatbot after tools
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
app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")
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