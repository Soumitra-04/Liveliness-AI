"""
routes/analyze.py
=================
Defines the POST /analyze endpoint.

Responsibility:
  1. Accept an uploaded file from the client.
  2. Delegate all heavy lifting to the service layer.
  3. Assemble and return a structured JSON response.

The router is registered in main.py under the /analyze prefix,
so the full path becomes POST /analyze.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.file_service import detect_file_type, save_file
from app.services.db_service import store_metadata

router = APIRouter()


@router.post(
    "/",
    summary="Analyze an uploaded media file",
    response_description="File metadata saved; ready for AI processing in future steps.",
)
async def analyze_file(file: UploadFile = File(...)):
    """
    **Step 1 — Infrastructure only** (no AI scoring yet).

    Flow
    ----
    1. Receive the uploaded file via multipart/form-data.
    2. Persist the file to storage via `save_file()`.
    3. Infer media type (image / video / audio) via `detect_file_type()`.
    4. Persist metadata (filename, type, path) via `store_metadata()`.
    5. Return a confirmation payload.

    Future steps will slot AI scoring between steps 4 and 5.
    """

    # ------------------------------------------------------------------
    # Step 1 — Validate that a file was actually received
    # ------------------------------------------------------------------
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # ------------------------------------------------------------------
    # Step 2 — Persist the raw file to disk / object storage
    #           save_file() returns the absolute path where it was saved.
    # ------------------------------------------------------------------
    file_path: str = await save_file(file)

    # ------------------------------------------------------------------
    # Step 3 — Detect media type from the filename extension
    #           Returns one of: "image" | "video" | "audio" | "unknown"
    # ------------------------------------------------------------------
    file_type: str = detect_file_type(file.filename)

    if file_type == "unknown":
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type for '{file.filename}'. "
                "Accepted: image (jpg/png/webp/gif), "
                "video (mp4/mov/avi/mkv), "
                "audio (mp3/wav/flac/ogg)."
            ),
        )

    # ------------------------------------------------------------------
    # Step 4 — Persist metadata to the database
    #           store_metadata() returns a dict with at least an "id" key.
    # ------------------------------------------------------------------
    metadata: dict = store_metadata(
        filename=file.filename,
        file_type=file_type,
        file_path=file_path,
    )

    # ------------------------------------------------------------------
    # Step 5 — Return structured confirmation response
    #           Future: authenticity_score, risk_level, explanation added here.
    # ------------------------------------------------------------------
    return {
        "id": metadata["id"],
        "filename": file.filename,
        "file_type": file_type,
        "status": "saved",
    }
