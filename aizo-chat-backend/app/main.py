# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import REDIS_URL, PORT
from app.core.graph import build_graph
from app import dependencies
from fastapi.staticfiles import StaticFiles
import os
# Import routers
from app.api.routes import chat, upload, threads
DOWNLOAD_DIR = "./downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize and cleanup resources."""
    
    try:
        # Initialize Redis client
        redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
        dependencies.set_redis(redis_client)
        
        # Initialize LangGraph with Redis checkpointer
        async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
            await checkpointer.setup()
            
            # Build and compile the graph
            workflow = build_graph()
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            dependencies.set_graph(compiled_graph)
            
            print(f"✅ Backend running on port {PORT}")
            yield
            
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        raise
        
    finally:
        # Cleanup
        redis = dependencies.get_redis()
        if redis:
            await redis.aclose()


# Create FastAPI app
app = FastAPI(
    title="Aizo Backend",
    description="Digital Transformation AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# ✅ ADD THIS LINE:
# This exposes the "downloads" folder to the web
app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")

# ... (Keep CORSMiddleware and other endpoints) ...
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(threads.router, tags=["Threads"])


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}