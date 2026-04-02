"""
fusion.py  —  Liveliness-AI  |  Multimodal Fusion Module
=========================================================
Combines image, video, and audio authenticity scores into a single
trust verdict using a fixed weighted average.

Weights
-------
  Image : 0.40   (richest single-frame evidence)
  Video : 0.30   (temporal consistency)
  Audio : 0.30   (vocal / acoustic analysis)

Risk classification (authenticity percentage)
---------------------------------------------
  70 – 100 %  →  LOW     (likely authentic)
  40 –  70 %  →  MEDIUM  (uncertain / suspicious)
   0 –  40 %  →  HIGH    (likely deepfake)
"""

from __future__ import annotations

from typing import Tuple, TypedDict

# ── Types ─────────────────────────────────────────────────────────────────────

ModalityResult = Tuple[float, str]
"""A (score, explanation) pair produced by any single-modality analyser.
score must be in [0.0, 1.0]; 0 = fake, 1 = real."""


class FusionOutput(TypedDict):
    authenticity_score:   float        # 0 – 100 percentage
    risk_classification:  str          # "LOW" | "MEDIUM" | "HIGH"
    flags:                list[str]    # non-empty explanations from each modality


# ── Configuration ─────────────────────────────────────────────────────────────

_WEIGHTS: dict[str, float] = {
    "image": 0.40,
    "video": 0.30,
    "audio": 0.30,
}

_RISK_BANDS: list[tuple[float, str]] = [
    # (lower_bound_inclusive, label)  — ordered highest first
    (70.0, "LOW"),
    (40.0, "MEDIUM"),
    (0.0,  "HIGH"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def combine_results(
    image_result: ModalityResult,
    video_result: ModalityResult,
    audio_result: ModalityResult,
) -> FusionOutput:
    """
    Merge per-modality scores into a final authenticity verdict.

    Parameters
    ----------
    image_result : (score: float, explanation: str)
    video_result : (score: float, explanation: str)
    audio_result : (score: float, explanation: str)
        Each score must be in [0.0, 1.0].
        Explanations that are empty / whitespace-only are silently ignored.

    Returns
    -------
    FusionOutput
        {
          "authenticity_score":  float,   # 0–100
          "risk_classification": str,     # "LOW" | "MEDIUM" | "HIGH"
          "flags":               list[str]
        }

    Example
    -------
    >>> result = combine_results(
    ...     image_result=(0.8, "No visual artefacts detected"),
    ...     video_result=(0.6, "Minor temporal inconsistencies"),
    ...     audio_result=(0.4, "Audio lacks natural variation"),
    ... )
    >>> result["authenticity_score"]
    62.0
    >>> result["risk_classification"]
    'MEDIUM'
    """
    modalities = {
        "image": image_result,
        "video": video_result,
        "audio": audio_result,
    }

    # ── Validate inputs ───────────────────────────────────────────────────────
    for name, (score, _) in modalities.items():
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"Score for '{name}' is {score!r} — must be in [0.0, 1.0]."
            )

    # ── Weighted average ──────────────────────────────────────────────────────
    raw_score: float = sum(
        _WEIGHTS[name] * score
        for name, (score, _) in modalities.items()
    )

    # Convert to 0–100 percentage, rounded to one decimal place
    authenticity_score = round(float(raw_score * 100), 1)
    authenticity_score = max(0.0, min(100.0, authenticity_score))  # clamp

    # ── Risk classification ───────────────────────────────────────────────────
    risk_classification = _classify_risk(authenticity_score)

    # ── Collect non-trivial explanations ─────────────────────────────────────
    flags: list[str] = [
        explanation.strip()
        for _, explanation in modalities.values()
        if explanation and explanation.strip()
    ]

    return FusionOutput(
        authenticity_score=authenticity_score,
        risk_classification=risk_classification,
        flags=flags,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_risk(pct: float) -> str:
    """Map an authenticity percentage to a risk label."""
    for lower_bound, label in _RISK_BANDS:
        if pct >= lower_bound:
            return label
    return "HIGH"  # fallback (should never be reached)