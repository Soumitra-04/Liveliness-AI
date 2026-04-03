"""
app/ai_engine/image_ela.py
==========================
Liveliness-AI — Hybrid Deepfake Detection (FINAL VERSION)

Features:
✔ HuggingFace pretrained model (ViT)
✔ Spatial + Noise signals
✔ Adaptive soft fusion (NO hardcoded thresholds)
✔ Robust error handling
✔ Explainable AI output
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------

_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

_CLASSIFIER = None
_LOAD_ERROR: Optional[str] = None


def _load_model() -> None:
    global _CLASSIFIER, _LOAD_ERROR

    try:
        from transformers import pipeline

        logger.info("Loading deepfake detection model...")

        _CLASSIFIER = pipeline(
            task="image-classification",
            model=_MODEL_ID,
        )

        logger.info("Model loaded successfully.")

    except ImportError:
        _LOAD_ERROR = "Transformers not installed. Run: pip install transformers torch"
        logger.error(_LOAD_ERROR)

    except Exception as e:
        _LOAD_ERROR = f"Model load failed: {e}"
        logger.error(_LOAD_ERROR)


# Load model at import
_load_model()


# ---------------------------------------------------------------------------
# MAIN PROCESS FUNCTION
# ---------------------------------------------------------------------------

def process_image(file_path: str) -> tuple[float, str]:
    """
    Hybrid deepfake detection:
    - Model prediction (HuggingFace)
    - Spatial analysis
    - Noise analysis
    - Adaptive fusion

    Returns:
        score (0–1): authenticity
        explanation (str)
    """

    # ------------------------------
    # Check model availability
    # ------------------------------
    if _CLASSIFIER is None:
        return 0.5, f"Error: {_LOAD_ERROR}"

    # ------------------------------
    # Load image safely
    # ------------------------------
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
        img = img.convert("RGB")

    except FileNotFoundError:
        return 0.5, f"Error: File not found — {file_path}"

    except UnidentifiedImageError:
        return 0.5, f"Error: Unsupported image format"

    except Exception as e:
        return 0.5, f"Error loading image: {e}"

    # ------------------------------
    # Run model inference
    # ------------------------------
    try:
        results = _CLASSIFIER(img)

    except Exception as e:
        return 0.5, f"Model inference error: {e}"

    # ------------------------------
    # Parse model output
    # ------------------------------
    real_score = None
    fake_score = None

    for r in results:
        label = r.get("label", "").lower()
        score = float(r.get("score", 0.0))

        if "real" in label:
            real_score = score
        elif "fake" in label:
            fake_score = score

    # Fallback handling
    if real_score is None and fake_score is None:
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top = sorted_results[0]

        if "real" in top["label"].lower():
            real_score = top["score"]
            fake_score = 1 - real_score
        else:
            fake_score = top["score"]
            real_score = 1 - fake_score

    if real_score is None:
        real_score = 1 - (fake_score or 0.5)

    if fake_score is None:
        fake_score = 1 - real_score

    # ------------------------------------------------------------------
    # 🔥 HYBRID SOFT FUSION (NO HARDCODED RULES)
    # ------------------------------------------------------------------

    spatial = None
    noise = None

    try:
        from app.ai_engine.spatial import spatial_inconsistency_score
        from app.ai_engine.noise import noise_score

        spatial = spatial_inconsistency_score(file_path)
        noise = noise_score(file_path)

        # Combine auxiliary signals
        aux_signal = (spatial + noise) / 2

        # Adaptive weighting (continuous logic)
        model_weight = real_score
        aux_weight = 1 - model_weight

        # Final score calculation
        final_score = (model_weight * real_score) + (aux_weight * (1 - aux_signal))

        score = round(float(final_score), 4)

    except Exception:
        # fallback if auxiliary modules fail
        score = round(float(real_score), 4)

    # ------------------------------
    # Build explanation
    # ------------------------------
    explanation = _build_explanation(score, real_score, fake_score)

    if spatial is not None and noise is not None:
        explanation += (
            f" | Spatial anomaly: {round(spatial,2)}, "
            f"Noise irregularity: {round(noise,2)}"
        )

    return score, explanation


# ---------------------------------------------------------------------------
# EXPLANATION FUNCTION
# ---------------------------------------------------------------------------

def _build_explanation(score: float, real_conf: float, fake_conf: float) -> str:
    """
    Generate human-readable explanation
    """

    real_pct = round(real_conf * 100, 1)
    fake_pct = round(fake_conf * 100, 1)

    if score >= 0.8:
        verdict = (
            "The image is highly likely to be authentic with no strong deepfake indicators."
        )
    elif score >= 0.6:
        verdict = (
            "The image appears mostly real, though minor inconsistencies may exist."
        )
    elif score >= 0.4:
        verdict = (
            "The model is uncertain. The image shows mixed characteristics of real and fake content."
        )
    elif score >= 0.2:
        verdict = (
            "The image likely contains manipulation or artificial generation artifacts."
        )
    else:
        verdict = (
            "The image is highly likely to be AI-generated or heavily manipulated."
        )

    return (
        f"{verdict} "
        f"(Model confidence → Real: {real_pct}% | Fake: {fake_pct}%)"
    )