# app/api/routes/upload.py
from fastapi import APIRouter, UploadFile, File

from app.services.file_processor import process_upload

router = APIRouter()


@router.post("/upload/{thread_id}")
async def upload_file(thread_id: str, file: UploadFile = File(...)):
    """Handle file uploads for a specific thread."""
    return await process_upload(file, thread_id)