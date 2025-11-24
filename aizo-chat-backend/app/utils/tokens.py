# app/utils/tokens.py
import tiktoken
from app.config import MAIN_MODEL


def count_tokens(text: str, model: str = MAIN_MODEL) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))