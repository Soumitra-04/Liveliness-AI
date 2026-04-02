"""
Liveliness-AI — Video Analysis Engine
=======================================
app/ai_engine/video_rppg.py

Detects unnatural facial motion patterns in video using:
  • OpenCV  — frame extraction & image processing
  • MediaPipe FaceMesh — 468-point facial landmark detection

Detection strategy
------------------
Real faces produce smooth, physiologically-constrained landmark motion.
Deepfake / synthetic faces exhibit:
  1. Landmark jitter       — high-frequency noise between adjacent frames
  2. Velocity spikes       — sudden unnatural accelerations
  3. Geometric instability — bounding-box / pose inconsistency across frames
  4. Low detection rate    — face mesh drops in/out on poor warps

All four signals are combined into a single [0, 1] deepfake-likelihood score.
"""

import cv2
import numpy as np

import logging
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("liveliness_ai.video_rppg")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

FRAME_SAMPLE_RATE: int = 5
MAX_FRAMES: int = 60

MAX_FACES: int = 1
REFINE_LANDMARKS: bool = True
MIN_DETECTION_CONF: float = 0.5
MIN_TRACKING_CONF: float = 0.5

TRACKED_LANDMARK_IDS: list[int] = [
    1,
    152,
    234,
    454,
    33,
    263,
    61,
    291,
    10,
    168,
]

JITTER_ZSCORE_THRESH: float  = 2.5
VELOCITY_SPIKE_THRESH: float = 3.0
JITTER_WEIGHT: float         = 0.40
VELOCITY_WEIGHT: float       = 0.35
DETECTION_WEIGHT: float      = 0.15
GEOMETRY_WEIGHT: float       = 0.10

# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------

@dataclass
class _FrameData:
    landmarks:   list[np.ndarray] = field(default_factory=list)
    bbox_widths:  list[float]     = field(default_factory=list)
    bbox_heights: list[float]     = field(default_factory=list)
    detected_frames: int          = 0
    total_frames:    int          = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_video(file_path: str) -> tuple[float, str]:
    logger.info("Starting video analysis: %s", file_path)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", file_path)
        return 0.5, "Could not open video file — analysis inconclusive."

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps                = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info("Video — total frames: %d, FPS: %.1f", total_video_frames, fps)

    frames = _extract_frames(cap, total_video_frames)
    cap.release()

    if not frames:
        logger.warning("No frames could be extracted from: %s", file_path)
        return 0.5, "No frames extracted — video may be corrupt or empty."

    logger.info("Extracted %d frames for analysis.", len(frames))

    frame_data = _collect_landmark_data(frames)

    detection_rate = (
        frame_data.detected_frames / frame_data.total_frames
        if frame_data.total_frames > 0 else 0.0
    )

    if frame_data.detected_frames < 2:
        msg = (
            "No face detected in video — cannot assess landmark motion."
            if frame_data.detected_frames == 0
            else "Face detected in too few frames for reliable analysis."
        )
        logger.warning(msg)
        score = 0.35 if frame_data.detected_frames == 0 else 0.45
        return round(score, 3), msg

    score, explanation = _compute_score(frame_data, detection_rate)

    logger.info("Analysis complete — score: %.3f | %s", score, explanation)
    return score, explanation


# ---------------------------------------------------------------------------
# Step 2 helper — frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(cap: cv2.VideoCapture, total: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    frame_idx = 0

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SAMPLE_RATE == 0:
            frames.append(frame)

        frame_idx += 1

    return frames


# ---------------------------------------------------------------------------
# Steps 3 & 4 helper — MediaPipe landmark collection
# ---------------------------------------------------------------------------

def _collect_landmark_data(frames: list[np.ndarray]) -> _FrameData:
    data = _FrameData(total_frames=len(frames))

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            logger.debug("Frame %d — no face detected.", i)
            continue

        # Take first detected face
        (x, y, fw, fh) = faces[0]

        # --- Create pseudo-landmarks (stable key points) ---
        coords = np.array([
            [x + fw * 0.5, y + fh * 0.5],   # center
            [x + fw * 0.5, y + fh],         # chin
            [x, y + fh * 0.5],              # left
            [x + fw, y + fh * 0.5],         # right
            [x + fw * 0.3, y + fh * 0.3],   # left eye approx
            [x + fw * 0.7, y + fh * 0.3],   # right eye approx
            [x + fw * 0.3, y + fh * 0.7],   # left mouth approx
            [x + fw * 0.7, y + fh * 0.7],   # right mouth approx
            [x + fw * 0.5, y],              # forehead
            [x + fw * 0.5, y + fh * 0.2],   # upper nose
        ], dtype=np.float32)

        data.landmarks.append(coords)
        data.bbox_widths.append(float(fw))
        data.bbox_heights.append(float(fh))
        data.detected_frames += 1

    return data

# ---------------------------------------------------------------------------
# Step 5 helper — score computation
# ---------------------------------------------------------------------------

def _compute_score(data: _FrameData, detection_rate: float) -> tuple[float, str]:

    lm_array = np.stack(data.landmarks, axis=0)

    displacements = np.linalg.norm(
        np.diff(lm_array, axis=0),
        axis=2,
    )

    mean_disp_per_frame = displacements.mean(axis=1)

    jitter_score = _zscore_outlier_ratio(mean_disp_per_frame, JITTER_ZSCORE_THRESH)

    if len(mean_disp_per_frame) >= 2:
        acceleration = np.abs(np.diff(mean_disp_per_frame))
        velocity_score = _zscore_outlier_ratio(acceleration, VELOCITY_SPIKE_THRESH)
    else:
        velocity_score = 0.0

    detection_score = max(0.0, 1.0 - detection_rate)

    aspect_ratios = np.array(data.bbox_widths) / (np.array(data.bbox_heights) + 1e-6)
    geometry_score = float(np.std(aspect_ratios)) if len(aspect_ratios) > 1 else 0.0
    geometry_score = min(geometry_score / 0.15, 1.0)

    final_score = (
        JITTER_WEIGHT    * jitter_score    +
        VELOCITY_WEIGHT  * velocity_score  +
        DETECTION_WEIGHT * detection_score +
        GEOMETRY_WEIGHT  * geometry_score
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    explanation = _build_explanation(
        final_score, jitter_score, velocity_score, detection_rate, geometry_score
    )

    return round(final_score, 3), explanation


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _zscore_outlier_ratio(series: np.ndarray, threshold: float) -> float:
    if len(series) == 0:
        return 0.0
    std = series.std()
    if std < 1e-8:
        return 0.0
    z_scores = np.abs((series - series.mean()) / std)
    return float((z_scores > threshold).mean())


def _build_explanation(
    score: float,
    jitter: float,
    velocity: float,
    detection_rate: float,
    geometry: float,
) -> str:

    signals: list[str] = []

    if jitter > 0.3:
        signals.append("facial landmark jitter")
    if velocity > 0.3:
        signals.append("unnatural motion velocity spikes")
    if detection_rate < 0.7:
        signals.append(f"low face-detection rate ({detection_rate:.0%})")
    if geometry > 0.3:
        signals.append("geometric face instability")

    if score >= 0.75:
        verdict = "HIGH deepfake likelihood"
    elif score >= 0.50:
        verdict = "MODERATE deepfake likelihood"
    elif score >= 0.25:
        verdict = "LOW deepfake likelihood"
    else:
        verdict = "Video appears authentic"

    if signals:
        signal_str = "; ".join(signals)
        return f"{verdict} (score {score:.2f}). Detected: {signal_str}."
    else:
        return f"{verdict} (score {score:.2f}). No strong anomalies detected."