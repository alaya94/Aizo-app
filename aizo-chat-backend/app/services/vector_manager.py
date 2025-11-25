import time
from collections import OrderedDict
from threading import Lock
from typing import Optional
import os 
DB_DIR = "./db"

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    """
    Thread-safe LRU cache for ChromaDB instances.
    Prevents memory leaks by:
    1. Limiting max stored instances (LRU eviction)
    2. Auto-cleanup of expired sessions
    3. Proper connection closing
    """
    
    def __init__(self, max_stores: int = 100, ttl_seconds: int = 3600):
        """
        Args:
            max_stores: Maximum number of ChromaDB instances to keep in memory
            ttl_seconds: Time-to-live for inactive sessions (default: 1 hour)
        """
        self.stores = OrderedDict()  # {session_id: ChromaDB instance}
        self.lock = Lock()           # Thread safety
        self.max_stores = max_stores
        self.ttl_seconds = ttl_seconds
        self.last_access = {}        # {session_id: timestamp}
        
        print(f"📦 VectorStoreManager initialized: max_stores={max_stores}, TTL={ttl_seconds}s")
    
    def get_store(self, session_id: str):
        """
        Get or create a ChromaDB instance for a session.
        Thread-safe with LRU eviction.
        
        IMPORTANT: This only manages IN-MEMORY instances.
        The actual vector data persists on disk and is reloaded if evicted.
        """
        with self.lock:
            # Case 1: Store exists - move to end (most recently used)
            if session_id in self.stores:
                self.stores.move_to_end(session_id)
                self.last_access[session_id] = time.time()
                print(f"♻️  Reusing cached store: {session_id}")
                return self.stores[session_id]
            
            # Case 2: Cache full - evict oldest (least recently used)
            if len(self.stores) >= self.max_stores:
                oldest_id, oldest_store = self.stores.popitem(last=False)
                
                # CRITICAL: Close connections before deletion
                # This only removes from MEMORY, disk data remains intact
                try:
                    # ChromaDB cleanup - close persistent connection
                    if hasattr(oldest_store, '_client'):
                        oldest_store._client.clear_system_cache()
                    print(f"🗑️  Evicted from cache: {oldest_id} (disk data preserved)")
                except Exception as e:
                    print(f"⚠️  Warning during eviction: {e}")
                
                del self.last_access[oldest_id]
            
            # Case 3: Load store (either new or from disk)
            path = os.path.join(DB_DIR, session_id)
            try:
                # ChromaDB automatically loads existing data if path exists
                store = Chroma(
                    persist_directory=path,
                    embedding_function=OpenAIEmbeddings(),
                    collection_name=f"collection_{session_id}"
                )
                self.stores[session_id] = store
                self.last_access[session_id] = time.time()
                
                # Check if this is a reload or new creation
                exists = os.path.exists(path) and os.listdir(path)
                action = "Reloaded" if exists else "Created new"
                print(f"✨ {action} store: {session_id} (total cached: {len(self.stores)})")
                
                return store
            except Exception as e:
                print(f"❌ Failed to load store for {session_id}: {e}")
                raise
    
    def cleanup_expired(self):
        """
        Remove stores that haven't been accessed in TTL seconds.
        Call this periodically from a background task.
        """
        now = time.time()
        expired_ids = []
        
        with self.lock:
            for session_id, last_access in list(self.last_access.items()):
                if now - last_access > self.ttl_seconds:
                    expired_ids.append(session_id)
            
            for session_id in expired_ids:
                if session_id in self.stores:
                    try:
                        store = self.stores[session_id]
                        if hasattr(store, '_client'):
                            store._client.clear_system_cache()
                        del self.stores[session_id]
                        del self.last_access[session_id]
                        print(f"🧹 Cleaned expired store: {session_id}")
                    except Exception as e:
                        print(f"⚠️  Error cleaning {session_id}: {e}")
        
        if expired_ids:
            print(f"📊 Cleanup complete: removed {len(expired_ids)} expired stores")
        
        return len(expired_ids)
    
    def get_stats(self) -> dict:
        """Get current cache statistics"""
        with self.lock:
            return {
                "active_stores": len(self.stores),
                "max_stores": self.max_stores,
                "sessions": list(self.stores.keys()),
                "oldest_access": min(self.last_access.values()) if self.last_access else None,
                "newest_access": max(self.last_access.values()) if self.last_access else None
            }