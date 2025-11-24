# app/services/file_processor.py
import os
from typing import List
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from app.services.vision import describe_image
from app.services.vector_store import get_vector_store


async def save_upload_file(file: UploadFile, thread_id: str) -> str:
    """Save uploaded file and return the path."""
    file_path = os.path.join(UPLOAD_DIR, f"{thread_id}_{file.filename}")
    
    file_size = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                os.remove(file_path)
                raise HTTPException(400, f"File too large. Limit is {MAX_FILE_SIZE_MB}MB.")
            buffer.write(chunk)
    
    return file_path


async def load_documents(file_path: str, ext: str, filename: str) -> List[Document]:
    """Load documents based on file extension."""
    docs = []
    
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
        print("   [DEBUG] Image detected. Analyzing with Vision Model...")
        description = await describe_image(file_path)
        docs = [Document(page_content=description, metadata={"source": filename})]
    
    return docs


async def process_upload(file: UploadFile, thread_id: str) -> dict:
    """Process an uploaded file and add to vector store."""
    print(f"\n📂 [DEBUG] Processing upload for Thread: {thread_id}")
    
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Save file
    file_path = await save_upload_file(file, thread_id)
    
    try:
        print(f"   [DEBUG] Loading {ext} file...")
        docs = await load_documents(file_path, ext, file.filename)
        
        # Handle scanned PDFs
        if ext == ".pdf" and docs and len(docs[0].page_content.strip()) < 10:
            print("⚠️ [DEBUG] PDF appears to be scanned (no text).")
            return {"status": "warning", "detail": "PDF contains no text (Scanned?)."}

        if not docs:
            return {"status": "error", "detail": "No content extracted."}

        # Split and store
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        ).split_documents(docs)
        
        store = get_vector_store(thread_id)
        store.add_documents(splits)
        print(f"✅ [DEBUG] Added {len(splits)} chunks to store.")
        
        return {"status": "success", "chunks": len(splits)}

    except Exception as e:
        print(f"❌ [DEBUG] Upload Error: {e}")
        raise HTTPException(500, f"Processing failed: {e}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)