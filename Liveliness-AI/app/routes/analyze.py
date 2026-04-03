"""
routes/analyze.py
=================
POST /analyze endpoint — full multimodal deepfake detection pipeline.

Processing flow
---------------
  1. Validate uploaded file
  2. Save file to disk
  3. Detect media type  (image / video / audio)
  4. Store metadata in database
  5. Route to the correct AI analyser:
       image → ELA + pretrained deepfake model   (image_ela.py)
       video → frame-based rPPG detection         (video_rppg.py)
       audio → Wav2Vec vocal analysis             (audio_vocal.py)
  6. Fuse per-modality scores into a single trust verdict (fusion.py)
  7. Attach metadata fields and return
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.file_service import detect_file_type, save_file
from app.services.db_service import store_metadata
from app.ai_engine.image_ela import process_image
from app.ai_engine.video_rppg import process_video
from app.ai_engine.audio_vocal import process_audio
from app.ai_engine.fusion import combine_results

router = APIRouter()


@router.post(
    "/",
    summary="Analyze an uploaded media file for deepfake indicators",
    response_description=(
        "Authenticity score (0–100), risk classification, and per-modality flags."
    ),
)
async def analyze_file(file: UploadFile = File(...)):

    # ── Step 1 — Validate file ────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # ── Step 2 — Save file ────────────────────────────────────────────────────
    file_path: str = await save_file(file)

    # ── Step 3 — Detect type ──────────────────────────────────────────────────
    file_type: str = detect_file_type(file.filename)

    if file_type == "unknown":
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type for '{file.filename}'. "
                "Accepted: image, video, audio."
            ),
        )

    # ── Step 4 — Store metadata ───────────────────────────────────────────────
    metadata: dict = store_metadata(
        filename=file.filename,
        file_type=file_type,
        file_path=file_path,
    )

    # ── Step 5 — AI Analysis ──────────────────────────────────────────────────

    # Default results for modalities not active in this request.
    # fusion.combine_results() requires all three to be present.
    image_result = (0.0, "No image processed")
    video_result = (0.0, "No video processed")
    audio_result = (0.0, "No audio processed")

    if file_type == "image":
        # ELA + pretrained deepfake model → (score, explanation)
        image_result = process_image(file_path)

    elif file_type == "video":
        # Frame-based rPPG detection → (score, explanation)
        video_result = process_video(file_path)

    elif file_type == "audio":
        # Wav2Vec vocal / acoustic analysis → (score, explanation)
        audio_result = process_audio(file_path)

    # ── Step 6 — Fuse modality scores into final verdict ──────────────────────
    final_output: dict = combine_results(
        image_result=image_result,
        video_result=video_result,
        audio_result=audio_result,
    )

    # ── Step 7 — Attach metadata and return ───────────────────────────────────
    final_output["id"]        = metadata["id"]
    final_output["filename"]  = metadata["filename"]
    final_output["file_type"] = metadata["file_type"]

    return final_output