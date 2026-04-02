"""
routes/analyze.py
=================
Enhanced POST /analyze endpoint with improved AI scoring.

Now includes:
  - FFT-based frequency analysis
  - Spatial inconsistency detection
  - Noise analysis
  - Combined scoring for better accuracy
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.file_service import detect_file_type, save_file
from app.services.db_service import store_metadata
from app.ai_engine.image_ela import process_image

# ✅ NEW IMPORTS
from app.ai_engine.spatial import spatial_inconsistency_score
from app.ai_engine.noise import noise_score

router = APIRouter()


@router.post(
    "/",
    summary="Analyze an uploaded media file",
    response_description="Returns authenticity score and explanation.",
)
async def analyze_file(file: UploadFile = File(...)):

    # ------------------------------
    # Step 1 — Validate file
    # ------------------------------
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # ------------------------------
    # Step 2 — Save file
    # ------------------------------
    file_path: str = await save_file(file)

    # ------------------------------
    # Step 3 — Detect type
    # ------------------------------
    file_type: str = detect_file_type(file.filename)

    if file_type == "unknown":
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type for '{file.filename}'. "
                "Accepted: image, video, audio."
            ),
        )

    # ------------------------------
    # Step 4 — Store metadata
    # ------------------------------
    metadata: dict = store_metadata(
        filename=file.filename,
        file_type=file_type,
        file_path=file_path,
    )

    # ------------------------------
    # Step 5 — AI Analysis
    # ------------------------------
    score = 0.5
    explanation = "Analysis not implemented for this type yet."

    if file_type == "image":
        # 🔥 Core FFT analysis
        fft_score, explanation = process_image(file_path)

        # 🔥 NEW: Spatial + Noise
        spatial_score = spatial_inconsistency_score(file_path)
        noise = noise_score(file_path)

        # 🔥 Combine scores (balanced weights)
        score = (
            0.5 * fft_score +
            0.3 * spatial_score +
            0.2 * noise
        )

        # 🔥 Improve explanation
        explanation += (
            f" | Spatial anomaly: {round(spatial_score,2)}, "
            f"Noise irregularity: {round(noise,2)}"
        )

        # 🔥 Boost suspicious cases
        if fft_score < 0.2 and spatial_score > 0.2:
            explanation += " However, possible structural inconsistencies detected."
            score = min(score + 0.2, 1.0)

    elif file_type == "video":
        score = 0.6
        explanation = "Video analysis pipeline initialized. Frame-level detection coming soon."

    elif file_type == "audio":
        score = 0.55
        explanation = "Audio analysis pipeline initialized. Voice anomaly detection coming soon."

    # ------------------------------
    # Step 6 — Risk Classification
    # ------------------------------
    if score > 0.7:
        risk = "High"
    elif score > 0.25:
        risk = "Medium"
    else:
        risk = "Low"

    # 🔥 Highlight suspicious cases
    if "structural inconsistencies" in explanation:
        explanation = "⚠️ " + explanation

    # ------------------------------
    # Step 7 — Response
    # ------------------------------
    return {
        "id": metadata["id"],
        "filename": metadata["filename"],
        "file_type": metadata["file_type"],
        "confidence": round(score * 100, 2),
        "risk": risk,
        "explanation": explanation,
    }