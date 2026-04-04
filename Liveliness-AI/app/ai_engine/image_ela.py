"""
app/ai_engine/image_ela.py
==========================
Liveliness-AI | Image Deepfake Detection — API-side processor

Pipeline (identical to test_inference.py)
-----------------------------------------
  1. PIL.Image.open(file_path).convert("RGB")
  2. T.Resize((224, 224))
  3. T.ToTensor()
  4. T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  5. model(tensor)   — inside torch.inference_mode()
  6. softmax → real_prob, fake_prob
  7. Forensic blend via screenshot_forensics.screenshot_resistant_score()

Model
-----
  DeepFakeV1Module (custom trained EfficientNet-V2-S + FrequencyStream)
  loaded via app.ai_engine.model_singleton (single global instance, eval()
  set at startup, never set to train() in the API path).
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def process_image(file_path: str) -> Tuple[float, str]:
    """
    Run deepfake detection on a single image file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a saved image (jpg / png / webp …).

    Returns
    -------
    (score, explanation) where
      score       : float in [0.0, 1.0]  —  1.0 = authentic, 0.0 = fake
      explanation : human-readable string describing what was found
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return 0.5, "Error: PyTorch not installed."

    # ── 1. Load model singleton ───────────────────────────────────────────────
    try:
        from app.ai_engine.model_singleton import get_model, get_device, preprocess_image_path
        model  = get_model()
        device = get_device()
    except RuntimeError as exc:
        logger.error("process_image: model not ready — %s", exc)
        return 0.5, f"Model not loaded: {exc}"

    # ── 2. Preprocess (MUST match test_inference.py exactly) ─────────────────
    try:
        tensor = preprocess_image_path(file_path).to(device)   # (1, 3, 224, 224)
    except Exception as exc:
        logger.error("process_image: preprocessing failed for %s — %s", file_path, exc)
        return 0.5, f"Image preprocessing error: {exc}"

    # ── 3. Forward pass — wrapped in inference_mode for speed + safety ────────
    # model.eval() was called globally in model_singleton.load_model().
    # torch.inference_mode() disables autograd tape for zero overhead.
    try:
        with torch.inference_mode():
            logits = model(tensor)                 # (1, 2)
        probs     = F.softmax(logits, dim=1)[0]   # (2,) on device
        fake_prob = float(probs[0].item())         # class 0 = fake
        real_prob = float(probs[1].item())         # class 1 = real
    except Exception as exc:
        logger.error("process_image: inference failed — %s", exc)
        return 0.5, f"Inference error: {exc}"

    logger.debug(
        "process_image: raw probs  fake=%.4f  real=%.4f  path=%s",
        fake_prob, real_prob, file_path,
    )

    # ── 4. Screenshot-resistant forensic blend ────────────────────────────────
    forensic_score   = 0.0
    forensic_detail  = {"ela": 0.0, "lbp": 0.0, "geometry": 0.0}
    forensic_applied = False
    adjusted_fake    = fake_prob * 100.0   # work in % for the blend

    try:
        from app.ai_engine.screenshot_forensics import screenshot_resistant_score
        f_score, f_detail = screenshot_resistant_score(file_path)
        forensic_score  = f_score
        forensic_detail = f_detail

        model_fake_pct = fake_prob * 100.0

        # Mirror the exact same blend logic as test_inference.py
        if f_score > 0.30 and model_fake_pct < 50.0:
            fake_target    = 85.0
            blend_amount   = f_score * (fake_target - model_fake_pct)
            adjusted_fake  = min(99.0, model_fake_pct + blend_amount)
            forensic_applied = adjusted_fake > 50.0
            logger.info(
                "process_image forensic blend: model_fake=%.1f%%  "
                "forensic=%.3f  adjusted=%.1f%%",
                model_fake_pct, f_score, adjusted_fake,
            )
        elif f_score > 0.20 and model_fake_pct >= 50.0:
            adjusted_fake = min(99.0, model_fake_pct + f_score * (99.0 - model_fake_pct) * 0.30)
            forensic_applied = True

        # Convert final adjusted fake% back to an authenticity score [0,1]
        final_real = max(0.0, min(1.0, 1.0 - adjusted_fake / 100.0))

    except Exception as exc:
        logger.warning("process_image: forensic analysis failed — %s", exc)
        final_real = real_prob   # fall back to raw model output

    # ── 5. Build explanation ──────────────────────────────────────────────────
    score       = round(float(final_real), 4)
    explanation = _build_explanation(
        score        = score,
        real_conf    = real_prob,
        fake_conf    = fake_prob,
        forensic_score   = forensic_score,
        forensic_detail  = forensic_detail,
        forensic_applied = forensic_applied,
    )

    return score, explanation


# ══════════════════════════════════════════════════════════════════════════════
# Explanation builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_explanation(
    score: float,
    real_conf: float,
    fake_conf: float,
    forensic_score: float   = 0.0,
    forensic_detail: dict   = {},
    forensic_applied: bool  = False,
) -> str:
    real_pct = round(real_conf * 100, 1)
    fake_pct = round(fake_conf * 100, 1)

    if score >= 0.80:
        verdict = "Highly authentic — no significant deepfake artefacts detected."
    elif score >= 0.60:
        verdict = "Likely authentic, though minor inconsistencies exist."
    elif score >= 0.40:
        verdict = "Uncertain — image shows mixed real/fake characteristics."
    elif score >= 0.20:
        verdict = "Likely manipulated — artificial generation artefacts detected."
    else:
        verdict = "Highly suspicious — strong deepfake or AI-generation evidence."

    base = (
        f"{verdict} "
        f"(Model → Real: {real_pct}% | Fake: {fake_pct}%)"
    )

    if forensic_applied:
        ela = round(forensic_detail.get("ela", 0.0) * 100, 1)
        lbp = round(forensic_detail.get("lbp", 0.0) * 100, 1)
        geo = round(forensic_detail.get("geometry", 0.0) * 100, 1)
        base += (
            f" | Forensic override applied (suspicion={forensic_score*100:.1f}%): "
            f"ELA={ela}% LBP={lbp}% Geo={geo}%"
        )
    elif forensic_score > 0.20:
        base += f" | Forensic signals mildly suspicious ({forensic_score*100:.0f}%)"

    return base