"""
app/ai_engine/image_ela.py
==========================
Liveliness-AI — Image Deepfake Feature Extraction
--------------------------------------------------
Technique: FFT-based Spectral Slope Analysis

Why FFT for deepfake detection?
--------------------------------
Real photographs obey a 1/f² power law: spatial frequency energy drops
steeply as frequency increases (slope ≈ −1.5 to −3.5 in log-log space).

AI-generated or manipulated images break this law because:
  ● GAN transposed-convolution checkerboard artifacts inject energy
    uniformly at all frequencies → slope flattens toward 0
  ● Inpainting / blending seams disrupt the natural roll-off at
    splice boundaries
  ● JPEG re-compression cycles add quantisation noise across the spectrum
  ● Upsampling pipelines introduce periodic ringing patterns

This module measures the spectral slope and supporting statistics
without any neural network — just Pillow + NumPy.

Algorithm
---------
1. Load image → grayscale → float64 array
2. 2-D FFT → shift zero-frequency to centre → log power spectrum
3. Compute radially-averaged power profile (ring bins)
4. Fit log(power) ~ slope·log(radius) + intercept via least squares
5. Map slope onto [0, 1] score; steeper slope = more authentic
"""

from __future__ import annotations

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Number of concentric rings used for the radial power profile.
_N_BINS: int = 32

# Spectral slope thresholds (empirically calibrated).
#
# Natural JPEG photographs: slope ≈ −1.5 to −3.5
# GAN / manipulated images:  slope ≈ −0.5 to  0.0
_SLOPE_AUTHENTIC: float = -1.6   # steeper → score 0.0
_SLOPE_FAKE: float = -0.4        # shallower → score 1.0

# Weight of each sub-signal in the final score
_SLOPE_WEIGHT: float = 0.75
_DEVIATION_WEIGHT: float = 0.25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_image(file_path: str) -> tuple[float, str]:
    """
    Analyse an image file for deepfake / manipulation artefacts using
    FFT spectral slope analysis.

    Parameters
    ----------
    file_path : str
        Path to the image file (JPEG, PNG, WebP, BMP, TIFF, …).

    Returns
    -------
    score : float
        Manipulation likelihood in [0.0, 1.0].
        0.0 = consistent with authentic photograph
        1.0 = strong evidence of AI generation or manipulation

    explanation : str
        Human-readable summary of the dominant finding.

    Notes
    -----
    Never raises — all exceptions are returned as (1.0, "Error: …").

    Examples
    --------
    >>> score, explanation = process_image("photo.jpg")
    >>> print(f"{score:.2f} — {explanation}")
    0.06 — Frequency profile consistent with a natural photograph. ...
    """

    # ------------------------------------------------------------------
    # Step 1 — Load image
    # ------------------------------------------------------------------
    try:
        img = Image.open(file_path)
        img.verify()            # detects truncated / corrupt headers
        img = Image.open(file_path)  # re-open; verify() exhausts the handle
    except FileNotFoundError:
        return 1.0, f"Error: file not found — '{file_path}'"
    except UnidentifiedImageError:
        return 1.0, f"Error: '{file_path}' is not a recognised image format."
    except Exception as exc:
        return 1.0, f"Error loading image: {exc}"

    # ------------------------------------------------------------------
    # Step 2 — Convert to grayscale
    #
    # Colour channels introduce inter-channel correlations that would
    # complicate frequency analysis.  A single luminance channel gives
    # a clean, unambiguous signal for structural frequency detection.
    # ------------------------------------------------------------------
    try:
        gray = img.convert("L")                   # "L" = 8-bit grayscale
        pixels = np.array(gray, dtype=np.float64)
    except Exception as exc:
        return 1.0, f"Error converting image to grayscale: {exc}"

    h, w = pixels.shape

    # Guard against degenerate images
    if h < 16 or w < 16:
        return 0.0, "Image too small to analyse (minimum 16×16 pixels)."

    # Guard against blank / solid-colour images — no frequency signal to measure
    if pixels.std() < 1e-6:
        return 0.0, "Image is blank or solid colour; no frequency analysis performed."

    # ------------------------------------------------------------------
    # Step 3 — Apply 2-D Fast Fourier Transform
    #
    # fft2  → complex frequency coefficients on a 2-D grid
    # fftshift → moves DC (zero-frequency) to the centre so the spectrum
    #            is a bullseye: centre = low freq, edges = high freq
    # ------------------------------------------------------------------
    fft_shifted = np.fft.fftshift(np.fft.fft2(pixels))

    # Log power spectrum — log1p avoids log(0) at the DC spike
    log_power = np.log1p(np.abs(fft_shifted) ** 2)

    # ------------------------------------------------------------------
    # Step 4 — Analyse frequency spectrum via radial power profile
    #
    # Build a normalised radial distance map from the image centre.
    # Bin log-power into concentric rings and average within each ring.
    # Fit log(ring_power) ~ slope · log(ring_radius) by least squares.
    # ------------------------------------------------------------------
    cy, cx = h / 2.0, w / 2.0
    y_idx, x_idx = np.ogrid[:h, :w]
    norm_radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2) / (min(h, w) / 2.0)

    # Radial ring averages (skip DC region near 0)
    bin_edges = np.linspace(0.02, 1.0, _N_BINS + 1)
    ring_power: list[float] = []
    ring_centres: list[float] = []

    for i in range(_N_BINS):
        mask = (norm_radius >= bin_edges[i]) & (norm_radius < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ring_power.append(log_power[mask].mean())
        ring_centres.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)

    ring_power_arr = np.array(ring_power)
    ring_log_r = np.log(np.array(ring_centres) + 1e-9)

    if len(ring_power_arr) < 4:
        return 0.0, "Image too small for reliable spectral analysis."

    # Least-squares fit: log(power) ≈ slope · log(radius) + intercept
    A = np.vstack([ring_log_r, np.ones_like(ring_log_r)]).T
    try:
        (slope, intercept), _, _, _ = np.linalg.lstsq(A, ring_power_arr, rcond=None)
    except np.linalg.LinAlgError:
        return 0.5, "Could not fit spectral profile; result is inconclusive."

    # ------------------------------------------------------------------
    # Step 5 — Detect abnormal high-frequency noise
    #
    # Primary: spectral slope
    #   Natural images → steep negative slope (large |slope|)  → low score
    #   Manipulated    → shallow slope (|slope| near 0)        → high score
    #
    # Secondary: residual deviation of actual profile from fitted line
    #   Smooth natural content → small residuals
    #   Patchwork / blending   → larger residuals at splice frequencies
    # ------------------------------------------------------------------

    # Map slope to [0, 1]: _SLOPE_AUTHENTIC → 0.0, _SLOPE_FAKE → 1.0
    slope_score = float(np.clip(
        (slope - _SLOPE_AUTHENTIC) / (_SLOPE_FAKE - _SLOPE_AUTHENTIC),
        0.0, 1.0,
    ))

    # Normalised residual std relative to the profile's dynamic range
    fitted = slope * ring_log_r + intercept
    residuals = ring_power_arr - fitted
    dynamic_range = ring_power_arr.max() - ring_power_arr.min() + 1e-9
    norm_residual = float(np.clip(residuals.std() / dynamic_range, 0.0, 1.0))

    # Weighted combination
    score = float(_SLOPE_WEIGHT * slope_score + _DEVIATION_WEIGHT * norm_residual)

    explanation = _build_explanation(score, slope, norm_residual)
    return round(score, 4), explanation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_explanation(score: float, slope: float, norm_residual: float) -> str:
    """Return a plain-English summary based on the computed metrics."""

    slope_str = f"{slope:.2f}"
    res_str = f"{norm_residual:.2f}"

    if score < 0.20:
        verdict = "Frequency profile consistent with a natural photograph."
        detail = (
            f"Spectral slope ({slope_str}) falls within the typical range for "
            "camera-captured imagery, indicating an authentic 1/f\u00b2 energy distribution."
        )
    elif score < 0.45:
        verdict = "Mild spectral anomalies detected; possible re-compression or light editing."
        detail = (
            f"Spectral slope ({slope_str}) is slightly shallower than expected. "
            "Consistent with JPEG re-encoding, sharpening filters, or moderate post-processing."
        )
    elif score < 0.70:
        verdict = "Unusual high-frequency patterns detected; possible splicing or partial AI generation."
        detail = (
            f"Spectral slope ({slope_str}) and residual deviation ({res_str}) suggest "
            "blending artefacts, GAN upsampling noise, or inpainting boundaries that "
            "disrupt the natural frequency roll-off."
        )
    else:
        verdict = "Strong manipulation artefacts present; likely AI-generated or heavily edited."
        detail = (
            f"Very shallow spectral slope ({slope_str}) and high residual deviation ({res_str}) "
            "are inconsistent with natural camera imagery. Frequency fingerprint matches "
            "patterns typical of GAN generation or heavy compositing."
        )

    return f"{verdict} {detail}"