# app/api/routes/threads.py
from fastapi import APIRouter

from app.services.vector_store import delete_vector_store
from app.dependencies import get_redis

router = APIRouter()


@router.delete("/delete_thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """Delete a thread and all associated data."""
    
    print(f"\n🗑️ [DEBUG] Request to delete thread: {thread_id}")
    
    # Delete Vector Store
    vs_deleted = delete_vector_store(thread_id)
    if vs_deleted:
        print("   ✅ Vector Store deleted.")
    else:
        print("   ⚠️ Vector Store not found.")

    # Delete Redis Memory
    redis = get_redis()
    if redis:
        try:
            await redis.delete(f"memory:{thread_id}")
            print("   ✅ Redis Memory deleted.")
        except Exception as e:
            print(f"   ❌ Error deleting Redis Memory: {e}")

    return {"status": "deleted", "thread_id": thread_id}