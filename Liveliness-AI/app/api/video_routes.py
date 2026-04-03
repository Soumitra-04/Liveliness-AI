"""
video_routes.py — FastAPI router for deepfake video detection.

Endpoint
--------
POST /detect-video
    Accept a video file upload, run analysis, return structured JSON.

Mount in main.py with:
    from app.api.video_routes import router as video_router
    app.include_router(video_router, prefix="/detect", tags=["Deepfake Detection"])
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.ai_engine.video_pipeline.file_handler import save_upload, validate_extension
from app.services.video_analyzer import analyze_video

router = APIRouter()


@router.post(
    "/detect-video",
    summary="Deepfake Video Detection",
    description=(
        "Upload a video file (.mp4, .avi, .mov, .mkv, .webm, .flv). "
        "The pipeline extracts frames at 2 FPS, detects faces, runs "
        "EfficientNet-B0 inference, and returns a structured verdict."
    ),
    response_description="Structured deepfake analysis result.",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file to analyse"),
):
    # ── Validate file extension ────────────────────────────────────────────────
    try:
        validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )

    # ── Save to temp, run pipeline, clean up ──────────────────────────────────
    try:
        async with save_upload(file) as tmp_path:
            result = analyze_video(tmp_path)
    except ValueError as exc:
        # No frames extracted
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Frame extraction failed: {exc}",
        )
    except RuntimeError as exc:
        # No faces detected
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Face detection failed: {exc}",
        )
    except Exception as exc:
        # Unexpected inference error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    return JSONResponse(content=result)
