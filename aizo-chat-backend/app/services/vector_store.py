# app/services/vector_store.py
import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import DB_DIR

# Cache for vector stores
_vector_stores: dict = {}


def get_vector_store(session_id: str) -> Chroma:
    """Get or create a vector store for a session."""
    path = os.path.join(DB_DIR, session_id)
    
    if session_id not in _vector_stores:
        _vector_stores[session_id] = Chroma(
            persist_directory=path,
            embedding_function=OpenAIEmbeddings(),
            collection_name=f"collection_{session_id}"
        )
    return _vector_stores[session_id]


def delete_vector_store(session_id: str) -> bool:
    """Delete a vector store for a session."""
    try:
        path = os.path.join(DB_DIR, session_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            if session_id in _vector_stores:
                del _vector_stores[session_id]
            return True
        return False
    except Exception as e:
        print(f"Error deleting vector store: {e}")
        return False