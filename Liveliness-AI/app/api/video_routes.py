"""
video_routes.py — FastAPI router for deepfake video detection.

Endpoint
--------
POST /detect/detect-video
    Accept a video file upload, run analysis, return structured JSON.

Detection pipeline (priority order)
------------------------------------
1. PRIMARY — rPPG (remote photoplethysmography) via video_rppg.process_video()
   Measures chrominance variance and cardiac-frequency SNR across 3 facial ROIs.
   Returns a score in [0, 1] where < ~0.40 → FAKE, > 0.65 → REAL.

2. FALLBACK — EfficientNet-B0 frame-level classifier via video_analyzer.analyze_video()
   Runs only if the rPPG pipeline raises an exception (e.g. missing face landmarks).
   Requires best_model-v3.pt weights to be present; returns UNCERTAIN if they
   are absent (rather than silently outputting wrong REAL verdicts).

Score convention (standardised in response)
-------------------------------------------
  fake_score : float [0, 1]  →  0 = certainly real, 1 = certainly fake
  verdict    : "FAKE" | "REAL" | "UNCERTAIN"
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _validate_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )
    return ext


@asynccontextmanager
async def _save_upload(upload: UploadFile) -> AsyncGenerator[str, None]:
    """Save an uploaded file to a temp location, yield its path, then delete it."""
    _, ext = os.path.splitext(upload.filename or "video.mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext.lower())
    try:
        while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
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


def _rppg_to_response(score: float, explanation: str) -> dict:
    """
    Convert rPPG (score, explanation) to the standardised response dict.

    rPPG score convention: high = REAL, low = FAKE.
    We invert to fake_score so response is consistent with image endpoint.
    """
    fake_score = round(1.0 - score, 4)

    if score >= 0.65:
        verdict = "REAL"
        confidence = "HIGH" if score >= 0.80 else "MEDIUM"
    elif score >= 0.40:
        verdict = "UNCERTAIN"
        confidence = "LOW"
    else:
        verdict = "FAKE"
        confidence = "HIGH" if score <= 0.20 else "MEDIUM"

    return {
        "verdict":     verdict,
        "fake_score":  fake_score,
        "confidence":  confidence,
        "pipeline":    "rPPG",
        "explanation": explanation,
    }


def _efficientnet_to_response(result: dict) -> dict:
    """
    Normalise the EfficientNet-pipeline result dict to the standardised format.
    The pipeline's confidence_score is already FAKE-probability.
    """
    verdict     = result.get("final_verdict", "UNCERTAIN")
    fake_score  = result.get("confidence_score", 0.5)
    metrics     = result.get("metrics", {})

    confidence = "LOW"
    if verdict in ("FAKE", "REAL"):
        margin = abs(fake_score - 0.5)
        confidence = "HIGH" if margin > 0.3 else "MEDIUM"

    return {
        "verdict":     verdict,
        "fake_score":  round(float(fake_score), 4),
        "confidence":  confidence,
        "pipeline":    "EfficientNet-B0",
        "explanation": (
            f"Frame-level analysis: "
            f"{metrics.get('frames_fake', '?')} / {metrics.get('frames_analysed', '?')} "
            f"frames flagged as fake. "
            f"Mean fake-score: {metrics.get('mean_score', 0):.3f}"
        ),
        "metrics": metrics,
    }


@router.post(
    "/detect-video",
    summary="Deepfake Video Detection",
    description=(
        "Upload a video file (.mp4, .avi, .mov, .mkv, .webm, .flv). "
        "Primary: rPPG cardiac-signal analysis across facial ROIs. "
        "Fallback: EfficientNet-B0 frame-level classifier."
    ),
    response_description="Structured deepfake analysis result.",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file to analyse"),
):
    # ── Validate extension ─────────────────────────────────────────────────────
    try:
        _validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )

    async with _save_upload(file) as tmp_path:

        # ── PRIMARY: rPPG pipeline ─────────────────────────────────────────────
        try:
            from app.ai_engine.video_rppg import process_video
            rppg_score, rppg_explanation = process_video(tmp_path)

            logger.info(
                "rPPG result: score=%.4f  explanation=%s",
                rppg_score, rppg_explanation,
            )
            print(
                f"[video_routes] rPPG → score={rppg_score:.4f}  "
                f"fake_score={1-rppg_score:.4f}  verdict="
                f"{'FAKE' if rppg_score < 0.40 else ('REAL' if rppg_score >= 0.65 else 'UNCERTAIN')}"
            )

            # rPPG returns specific strings when it cannot process the video
            # (e.g. no file, no FaceLandmarker model, or no face detected).
            # For these cases, fall through to the EfficientNet pipeline.
            fallback_triggers = [
                "File not found",
                "FaceLandmarker model missing",
                "Could not open video file",
                "Insufficient face data detected",
            ]
            
            if any(trigger in rppg_explanation for trigger in fallback_triggers):
                raise RuntimeError(
                    f"rPPG could not process video (score={rppg_score}: {rppg_explanation})"
                )

            return JSONResponse(content=_rppg_to_response(rppg_score, rppg_explanation))

        except (ImportError, RuntimeError) as rppg_err:
            logger.warning(
                "rPPG pipeline failed (%s) — falling back to EfficientNet.", rppg_err
            )

        # ── FALLBACK: EfficientNet-B0 frame-level pipeline ─────────────────────
        try:
            from app.services.video_analyzer import analyze_video
            result = analyze_video(tmp_path)
            logger.info("EfficientNet result: %s", result.get("final_verdict"))
            return JSONResponse(content=_efficientnet_to_response(result))

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Frame extraction failed: {exc}",
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Face detection failed: {exc}",
            )
        except Exception as exc:
            logger.exception("Both video pipelines failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video analysis error: {exc}",
            )
