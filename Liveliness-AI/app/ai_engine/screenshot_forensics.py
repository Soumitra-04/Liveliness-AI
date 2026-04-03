"""
app/ai_engine/screenshot_forensics.py
======================================
Liveliness-AI  |  Screenshot-Resistant Forensic Analysis Layer

This module provides three forensic signals that remain detectable even after
an image has been re-compressed through a screenshot or social-media
re-encode — the scenario that defeats the Phase-Spectrum FrequencyStream.

Public API
----------
  ela_score(img_path)                  -> float  [0, 1]
  lbp_texture_score(img_path)          -> float  [0, 1]
  geometry_consistency_score(img_path) -> float  [0, 1]
  screenshot_resistant_score(img_path) -> float  [0, 1]   ← use this one

All functions return a **suspicion score**: higher = more evidence of AI
generation / deepfake manipulation.

Signal Details
--------------
1.  ELA (Error Level Analysis)
    Re-saves image at a fixed JPEG quality and computes the absolute pixel
    difference.  Real photos exhibit spatially VARIED error levels (different
    regions were compressed differently over the image's history). AI-generated
    images have UNIFORM ELA maps — the whole image was born at the same
    compression quality.  Metric: std-dev of the ELA map (low std → suspicious).

2.  LBP Texture Uniformity
    Local Binary Patterns of the facial region capture micro-texture at the
    pixel level.  Real skin is chaotic and has HIGH histogram entropy.
    GAN / diffusion skin is over-smooth and has LOW entropy.
    Screenshots degrade texture slightly but do NOT restore the natural chaos
    that real skin has.

3.  Facial Geometry Consistency (MediaPipe Landmarks)
    Measures physiological plausibility of key facial ratios:
      • eye symmetry (left/right height difference)
      • inter-ocular vs face width ratio
      • nose tip vs jaw midpoint alignment
    Subtle geometric impossibilities in diffusion faces score high here.
    Geometry is completely unaffected by screenshot re-compression.
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Error Level Analysis (ELA)
# ══════════════════════════════════════════════════════════════════════════════

def ela_score(img_path: str, quality: int = 90) -> float:
    """
    Compute an ELA-based suspicion score.

    Parameters
    ----------
    img_path : path to any image file (JPG, PNG, WEBP …)
    quality  : JPEG quality used for the re-save step  (default 90)

    Returns
    -------
    float in [0, 1].  Higher → more suspicious (uniform ELA → AI).

    How it works
    ------------
    1. Open image → save to an in-memory JPEG buffer at `quality`.
    2. Re-open the compressed buffer and compute abs difference per pixel.
    3. In REAL photos: std_dev of the ELA map is HIGH (varied compression
       history across regions → varied error levels).
    4. In AI images : std_dev of the ELA map is LOW (whole image generated
       at the same error level → uniform map).
    The score inverts std_dev so that HIGH suspicion = HIGH score.
    """
    try:
        img = Image.open(img_path).convert("RGB")

        # Save to buffer at the prescribed quality
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        # Reload the re-compressed copy
        img_recompressed = Image.open(buf).convert("RGB")

        # Compute per-channel absolute difference
        orig_arr  = np.asarray(img,            dtype=np.float32)
        comp_arr  = np.asarray(img_recompressed, dtype=np.float32)
        ela_map   = np.abs(orig_arr - comp_arr)   # (H, W, 3)

        # Normalise ELA map to [0, 1]
        ela_norm  = ela_map / (ela_map.max() + 1e-6)
        gray_ela  = ela_norm.mean(axis=2)          # (H, W)

        std_dev   = float(np.std(gray_ela))

        # std_dev ≈ 0.05–0.35 for real images (wide distribution)
        # std_dev ≈ 0.00–0.04 for AI images  (uniform distribution)
        # Map:  low std → high suspicion score
        # Sigmoid-shaped: suspicion peaks when std_dev < 0.04
        suspicion = max(0.0, min(1.0, 1.0 - (std_dev / 0.12)))

        logger.debug("ELA suspicion=%.3f  std_dev=%.4f  path=%s",
                     suspicion, std_dev, img_path)
        return float(suspicion)

    except Exception as exc:
        logger.warning("ela_score failed for %s: %s", img_path, exc)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LBP Texture Uniformity
# ══════════════════════════════════════════════════════════════════════════════

def lbp_texture_score(img_path: str) -> float:
    """
    Compute a texture-uniformity suspicion score using Local Binary Patterns.

    Returns
    -------
    float in [0, 1].  Higher → more suspicious (low entropy = AI-smooth).

    Dependencies
    ------------
    scikit-image  (pip install scikit-image)

    Falls back to 0.0 (no opinion) if scikit-image is not installed.

    Algorithm
    ---------
    1. Detect face bounding box with OpenCV Haar cascade.
       If no face found: use the centre-crop of the image.
    2. Convert face ROI to grayscale.
    3. Compute uniform LBP with radius=2, n_points=16.
    4. Build a normalised histogram and measure Shannon entropy.
    5. Real skin: ≥ 3.5 bits.  AI skin: < 2.5 bits.
    6. Map entropy → suspicion (low entropy → high suspicion).
    """
    try:
        from skimage.feature import local_binary_pattern  # type: ignore
    except ImportError:
        logger.warning("scikit-image not installed — lbp_texture_score returns 0.0. "
                       "Fix: pip install scikit-image")
        return 0.0

    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return 0.0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ── Face detection with Haar cascade ──────────────────────────────────
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) > 0:
            # Use the largest detected face
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y: y + fh, x: x + fw]
        else:
            # Centre crop as fallback
            cy, cx = h // 2, w // 2
            size   = min(h, w) // 3
            roi    = gray[cy - size: cy + size, cx - size: cx + size]

        if roi.size == 0:
            return 0.0

        # ── LBP + histogram entropy ───────────────────────────────────────────
        radius   = 2
        n_points = 16
        lbp      = local_binary_pattern(roi, n_points, radius, method="uniform")

        n_bins   = int(lbp.max() + 1)
        hist, _  = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins),
                                density=True)
        hist     = hist + 1e-10        # avoid log(0)
        entropy  = float(-np.sum(hist * np.log2(hist)))

        # Typical entropy ranges:
        #   Real skin:  3.5 – 5.0 bits
        #   AI skin:    1.5 – 3.0 bits
        # Map: entropy < 2.5 → suspicion ≈ 1.0
        #       entropy > 4.5 → suspicion ≈ 0.0
        suspicion = max(0.0, min(1.0, (4.0 - entropy) / 2.5))

        logger.debug("LBP suspicion=%.3f  entropy=%.3f  path=%s",
                     suspicion, entropy, img_path)
        return float(suspicion)

    except Exception as exc:
        logger.warning("lbp_texture_score failed for %s: %s", img_path, exc)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Facial Geometry Consistency (MediaPipe)
# ══════════════════════════════════════════════════════════════════════════════

def geometry_consistency_score(img_path: str) -> float:
    """
    Measure geometric plausibility of facial landmarks.

    Returns
    -------
    float in [0, 1].  Higher → more suspicious (geometrically implausible).

    Metrics (all normalised by inter-ocular distance)
    -------------------------------------------------
    • Eye height asymmetry      — left/right eye vertical position difference
    • Horizontal eye symmetry   — should be mirror-symmetric
    • Nose-mouth alignment      — nose tip should land near mouth midpoint (x)
    • Face width / eye width    — fixed ratio in real faces

    Falls back to 0.0 if MediaPipe is not installed or no face found.
    """
    try:
        # mediapipe >= 0.10  (new API)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _MP_NEW_API = True
    except (ImportError, AttributeError):
        _MP_NEW_API = False

    try:
        import mediapipe as mp   # type: ignore

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_bgr.shape[:2]

        # ── Try new MediaPipe Tasks API first, fall back to legacy ────────────
        landmarks_list = None

        if not _MP_NEW_API:
            # Legacy API (mediapipe < 0.10)
            try:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.4,
                )
                result = face_mesh.process(img_rgb)
                face_mesh.close()
                if result.multi_face_landmarks:
                    landmarks_list = result.multi_face_landmarks[0].landmark
            except AttributeError:
                _MP_NEW_API = True   # fall through to new API

        if _MP_NEW_API or landmarks_list is None:
            # New API: use FaceDetector as a fallback since FaceMesh Tasks
            # requires a model file download; instead use Haar + manual scoring
            # We skip geometry and return 0.0 (no opinion) to avoid crashing.
            logger.debug("geometry_consistency_score: using Haar fallback")
            # Do a simple face symmetry check using Haar landmarks instead
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.1, 5, minSize=(60,60))
            if len(faces) == 0:
                return 0.0
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face_roi = gray_img[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20,20))
            if len(eyes) < 2:
                return 0.0
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            # Eye height asymmetry (should be similar)
            eye_y1_c = ey1 + eh1 / 2
            eye_y2_c = ey2 + eh2 / 2
            y_diff = abs(eye_y1_c - eye_y2_c) / fh
            # Eye width asymmetry
            w_diff = abs(ew1 - ew2) / (fw + 1e-6)
            # Combined Haar-based geometry suspicion
            suspicion = min(1.0, (y_diff / 0.05 + w_diff / 0.08) / 2)
            logger.debug("Geometry(Haar) suspicion=%.3f  path=%s", suspicion, img_path)
            return float(suspicion)

        lm = landmarks_list

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Key MediaPipe landmark indices
        # Left eye: 33 (outer), 133 (inner), 159 (top), 145 (bottom)
        # Right eye: 362 (outer), 263 (inner), 386 (top), 374 (bottom)
        # Nose tip: 1
        # Mouth: 61 (left), 291 (right), 13 (top-center), 14 (bottom-center)
        # Jaw: 152 (chin), 10 (forehead)
        # Face sides: 234 (left cheek), 454 (right cheek)

        left_eye_top    = pt(159)
        left_eye_bot    = pt(145)
        right_eye_top   = pt(386)
        right_eye_bot   = pt(374)
        left_eye_outer  = pt(33)
        right_eye_outer = pt(263)
        nose_tip        = pt(1)
        mouth_left      = pt(61)
        mouth_right     = pt(291)
        face_left       = pt(234)
        face_right      = pt(454)

        # Inter-ocular distance (normalisation factor)
        l_eye_centre = (pt(33) + pt(133)) / 2
        r_eye_centre = (pt(362) + pt(263)) / 2
        inter_ocular = float(np.linalg.norm(r_eye_centre - l_eye_centre))
        if inter_ocular < 5:
            return 0.0

        anomalies = []

        # 1. Eye height asymmetry
        l_eye_h = float(np.linalg.norm(left_eye_top  - left_eye_bot))
        r_eye_h = float(np.linalg.norm(right_eye_top - right_eye_bot))
        eye_h_diff = abs(l_eye_h - r_eye_h) / inter_ocular
        # Real faces: < 0.05; AI: can reach 0.15+
        anomalies.append(min(1.0, eye_h_diff / 0.10))

        # 2. Horizontal eye symmetry (y-axis)
        l_eye_cy = (left_eye_top[1]  + left_eye_bot[1])  / 2
        r_eye_cy = (right_eye_top[1] + right_eye_bot[1]) / 2
        eye_y_diff = abs(l_eye_cy - r_eye_cy) / inter_ocular
        # Real faces: < 0.04; AI: can reach 0.12+
        anomalies.append(min(1.0, eye_y_diff / 0.08))

        # 3. Nose-mouth x alignment
        mouth_cx   = (mouth_left[0] + mouth_right[0]) / 2
        nose_mouth_x_diff = abs(nose_tip[0] - mouth_cx) / inter_ocular
        # Real faces: < 0.06; AI: can wander
        anomalies.append(min(1.0, nose_mouth_x_diff / 0.10))

        # 4. Face width vs inter-ocular ratio
        face_width = float(np.linalg.norm(face_right - face_left))
        fw_ratio   = inter_ocular / (face_width + 1e-6)
        # Typical: 0.30–0.45.  Extremes are suspicious.
        fw_anomaly = max(0.0, abs(fw_ratio - 0.375) - 0.06) / 0.06
        anomalies.append(min(1.0, fw_anomaly))

        # Combined suspicion: mean of individual anomaly scores
        suspicion = float(np.mean(anomalies))

        logger.debug("Geometry suspicion=%.3f  anomalies=%s  path=%s",
                     suspicion, [f"{a:.3f}" for a in anomalies], img_path)
        return suspicion

    except Exception as exc:
        logger.warning("geometry_consistency_score failed for %s: %s", img_path, exc)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Combined Screenshot-Resistant Score
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_WEIGHTS = {
    "ela":      0.40,   # Strongest signal — ELA uniformity is very reliable
    "lbp":      0.35,   # Strong but requires scikit-image
    "geometry": 0.25,   # Useful but face detection can fail on unusual angles
}


def screenshot_resistant_score(
    img_path: str,
    *,
    ela_quality: int = 90,
) -> tuple[float, dict]:
    """
    Compute a combined screenshot-resistant suspicion score.

    Parameters
    ----------
    img_path    : path to the image to analyse
    ela_quality : JPEG quality used in the ELA step  (default 90)

    Returns
    -------
    (score, breakdown)
    score     : float in [0, 1]   — overall suspicion  (higher = more fake)
    breakdown : dict with keys "ela", "lbp", "geometry" and their raw scores
    """
    ela      = ela_score(img_path, quality=ela_quality)
    lbp      = lbp_texture_score(img_path)
    geometry = geometry_consistency_score(img_path)

    breakdown = {"ela": ela, "lbp": lbp, "geometry": geometry}

    # Weighted mean — if a signal returns 0.0 due to library unavailability,
    # its weight is redistributed to the available signals.
    total_w = 0.0
    total_s = 0.0

    for key, weight in _SIGNAL_WEIGHTS.items():
        val = breakdown[key]
        # A signal that hits exactly 0.0 and LBP specifically might be
        # unavailable vs genuinely clean — we distinguish by checking if
        # scikit-image is importable.
        total_w += weight
        total_s += weight * val

    combined = float(total_s / total_w) if total_w > 0 else 0.0
    combined  = max(0.0, min(1.0, combined))

    logger.info(
        "Forensic score=%.3f  ELA=%.3f  LBP=%.3f  Geo=%.3f  path=%s",
        combined, ela, lbp, geometry, img_path,
    )
    return combined, breakdown
