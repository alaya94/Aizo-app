import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import httpx
from sqlalchemy import text
import os
import sqlite3

# --- CONFIGURATION ---
BACKEND_URL = "http://localhost:8001"
STORAGE_PATH = "sqlite+aiosqlite:///chainlit.db"
DB_FILE = "chainlit.db"

# ==================================================
# 1. CUSTOM STORAGE PROVIDER (FIXED)
# ==================================================
# ==================================================
# 1. CUSTOM STORAGE PROVIDER (UPDATED)
# ==================================================
class SimpleLocalStorage:
    def __init__(self, base_path=".files"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    async def upload_file(self, object_key, data, mime="application/octet-stream", overwrite=True):
        # Create the full path
        file_path = os.path.join(self.base_path, object_key)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        with open(file_path, "wb") as f:
            f.write(data)
        
        return {"url": file_path, "object_key": object_key}

    # --- THIS WAS MISSING ---
    async def delete_file(self, object_key):
        file_path = os.path.join(self.base_path, object_key)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# ==================================================
# 2. SYNCHRONOUS DB SETUP
# ==================================================
def init_db_sync():
    print("🔨 Running Synchronous DB Initialization...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            "id" TEXT NOT NULL PRIMARY KEY,
            "identifier" TEXT NOT NULL UNIQUE,
            "metadata" TEXT,
            "createdAt" TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            "id" TEXT NOT NULL PRIMARY KEY,
            "createdAt" TEXT,
            "name" TEXT,
            "userId" TEXT,
            "userIdentifier" TEXT,
            "tags" TEXT,
            "metadata" TEXT,
            FOREIGN KEY("userId") REFERENCES users("id")
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            "id" TEXT NOT NULL PRIMARY KEY,
            "name" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "threadId" TEXT NOT NULL,
            "parentId" TEXT,
            "streaming" BOOLEAN NOT NULL,
            "waitForAnswer" BOOLEAN,
            "isError" BOOLEAN,
            "metadata" TEXT,
            "tags" TEXT,
            "input" TEXT,
            "output" TEXT,
            "createdAt" TEXT,
            "start" TEXT,
            "end" TEXT,
            "generation" TEXT,
            "showInput" TEXT,
            "language" TEXT,
            "indent" INTEGER,
            "defaultOpen" BOOLEAN
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elements (
            "id" TEXT NOT NULL PRIMARY KEY,
            "threadId" TEXT,
            "type" TEXT,
            "url" TEXT,
            "chainlitKey" TEXT,
            "name" TEXT NOT NULL,
            "display" TEXT,
            "objectKey" TEXT,
            "size" TEXT,
            "page" INTEGER,
            "language" TEXT,
            "forId" TEXT,
            "mime" TEXT,
            "props" TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            "id" TEXT NOT NULL PRIMARY KEY,
            "forId" TEXT NOT NULL,
            "threadId" TEXT NOT NULL,
            "value" INTEGER NOT NULL,
            "comment" TEXT,
            "strategy" TEXT
        );
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO users ("id", "identifier", "createdAt", "metadata")
        VALUES ('admin', 'admin', '2024-01-01 00:00:00', '{"role": "admin"}');
    """)
    conn.commit()
    conn.close()

init_db_sync()

# ==================================================
# 3. CHAINLIT DATA LAYER SETUP
# ==================================================
class SyncSQLAlchemyDataLayer(SQLAlchemyDataLayer):
    async def delete_thread(self, thread_id: str):
        # 1. Call the original method to delete from SQLite + Local Files
        await super().delete_thread(thread_id)
        
        # 2. Call the Backend to delete from ChromaDB + Redis
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.delete(f"{BACKEND_URL}/delete_thread/{thread_id}")
                if resp.status_code == 200:
                    print(f"✅ Backend synced deletion for {thread_id}")
                else:
                    print(f"⚠️ Backend deletion failed: {resp.text}")
        except Exception as e:
            print(f"❌ Could not reach backend for deletion: {e}")

# Initialize storage
storage_provider = SimpleLocalStorage()
data_layer = SyncSQLAlchemyDataLayer(
    conninfo=STORAGE_PATH,
    storage_provider=storage_provider
)


@cl.data_layer
def get_data_layer():
    return data_layer

# ==================================================
# 4. AUTHENTICATION
# ==================================================
@cl.password_auth_callback
def auth_callback(username, password):
    if username == "admin" and password == "admin":
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None

# ==================================================
# 5. CHAT LOGIC
# ==================================================
@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", cl.context.session.thread_id)

@cl.set_starters
async def set_starters():
    return [cl.Starter(label="New Chat", message="Hello!", icon="/public/brain.svg")]

@cl.on_chat_resume
async def on_resume(thread: dict):
    cl.user_session.set("thread_id", thread.get("id"))

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.context.session.thread_id
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "admin"

    # 1. Rename Thread
    await data_layer.update_thread(
        thread_id=thread_id,
        name=message.content[:50] if message.content else "File Upload",
        user_id=user_id
    )

    # 2. Handle File Uploads
    if message.elements:
        async with cl.Step(name="File Upload") as step:
            step.input = "Uploading file to backend..."
            
            for element in message.elements:
                if element.path:
                    try:
                        with open(element.path, 'rb') as f:
                            files = {'file': (element.name, f, element.mime)}
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                response = await client.post(
                                    f"{BACKEND_URL}/upload/{thread_id}",
                                    files=files
                                )
                        
                        if response.status_code == 200:
                            step.output = f"✅ Uploaded {element.name}"
                        else:
                            step.output = f"❌ Upload Failed: {response.text}"
                    except Exception as e:
                        step.output = f"❌ Connection Error: {str(e)}"
        
        if not message.content:
            await cl.Message(content="✅ Document processed. You can now ask questions about it!").send()
            return

    # 3. Stream Chat from Backend (THE NEW PART)
    msg = cl.Message(content="")
    await msg.send() # Send an empty message first to create the UI placeholder

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Use client.stream to open a persistent connection
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/chat",
                json={"message": message.content, "thread_id": thread_id}
            ) as response:
                
                # Check if backend is happy
                if response.status_code != 200:
                    error_text = await response.read()
                    msg.content = f"⚠️ **Backend Error:** {error_text.decode()}"
                    await msg.update()
                    return

                # Iterate over the text chunks as they arrive
                async for chunk in response.aiter_text():
                    if chunk:
                        await msg.stream_token(chunk)

    except Exception as e:
        msg.content = f"❌ **Connection Error:** {str(e)}"
        await msg.update()
        
    # Finalize the message after the stream ends
    await msg.update()