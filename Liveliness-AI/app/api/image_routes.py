"""
image_routes.py — FastAPI router for deepfake image detection.

Endpoint
--------
POST /detect-image
    Accept an image file upload, run analysis, return structured JSON.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.ai_engine.image_ela import process_image

router = APIRouter()

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )
    return ext


@asynccontextmanager
async def save_upload(upload: UploadFile) -> AsyncGenerator[str, None]:
    _, ext = os.path.splitext(upload.filename or "image.jpg")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext.lower())
    try:
        while chunk := await upload.read(1024 * 1024):
            tmp.write(chunk)
        tmp.close()
        yield tmp.name
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass


@router.post(
    "/detect-image",
    summary="Deepfake Image Detection",
    description=(
        "Upload an image file (.jpg, .jpeg, .png, .webp). "
        "The pipeline runs ELA + a pretrained huggingface classifier, "
        "and returns a structured verdict."
    ),
    response_description="Structured deepfake analysis result.",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyse"),
):
    try:
        validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )

    try:
        async with save_upload(file) as tmp_path:
            # process_image returns (authenticity_score, explanation)
            # authenticity_score: 1.0 = Real, 0.0 = Fake
            auth_score, explanation = process_image(tmp_path)
            
            # standardize to fake_score (0.0 to 1.0)
            fake_score = 1.0 - auth_score
            verdict = "FAKE" if fake_score > 0.50 else "REAL"

            result = {
                "verdict": verdict,
                "fake_score": round(fake_score, 4),
                "explanation": explanation
            }
            
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    return JSONResponse(content=result)
