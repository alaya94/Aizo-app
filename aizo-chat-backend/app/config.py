# app/config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- SERVER CONFIG ---
PORT = int(os.environ.get("PORT", 8001))

# --- REDIS CONFIG ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# --- DIRECTORIES ---
UPLOAD_DIR = "./uploads"
DB_DIR = "./db"

# --- FILE UPLOAD LIMITS ---
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

# --- LLM MODELS ---
MAIN_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
ROUTER_MODEL = "gpt-4o-mini"

# --- MEMORY SETTINGS ---
MAX_PROFILE_TOKENS = 1000
SUMMARY_THRESHOLD = 6  # Messages before summarization kicks in

# Create directories on import
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)