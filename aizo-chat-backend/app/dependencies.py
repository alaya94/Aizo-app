# app/dependencies.py
"""
Shared dependencies and global state management.
"""
from redis.asyncio import Redis

# Global instances (initialized in lifespan)
redis_client: Redis | None = None
graph = None


def get_redis() -> Redis | None:
    """Get the Redis client instance."""
    return redis_client


def get_graph():
    """Get the compiled LangGraph instance."""
    return graph


def set_redis(client: Redis):
    """Set the Redis client instance."""
    global redis_client
    redis_client = client


def set_graph(compiled_graph):
    """Set the compiled graph instance."""
    global graph
    graph = compiled_graph