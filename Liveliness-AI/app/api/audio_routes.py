"""
audio_routes.py — FastAPI router for deepfake audio detection.

Endpoint
--------
POST /detect-audio
    Accept an audio file upload, run analysis, return structured JSON.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.ai_engine.audio_vocal import process_audio

router = APIRouter()

_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}

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
    _, ext = os.path.splitext(upload.filename or "audio.wav")
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
    "/detect-audio",
    summary="Deepfake Audio Detection",
    description=(
        "Upload an audio file (.mp3, .wav, .m4a, etc). "
        "The pipeline runs temporal windows analysis through librosa/torch, "
        "and returns a structured verdict."
    ),
    response_description="Structured deepfake analysis result.",
)
async def detect_audio(
    file: UploadFile = File(..., description="Audio file to analyse"),
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
            # process_audio returns (score, explanation)
            # score: 0.0 = human, 1.0 = synthetic
            raw_fake_score, explanation = process_audio(tmp_path)
            
            # The score is already essentially a fake_score
            verdict = "FAKE" if raw_fake_score > 0.50 else "REAL"

            result = {
                "verdict": verdict,
                "fake_score": round(raw_fake_score, 4),
                "explanation": explanation
            }
            
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    return JSONResponse(content=result)
