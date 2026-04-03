"""
ai_engine/video_rppg.py  —  Liveliness-AI
==========================================
Rebuilt Production-grade rPPG deepfake detector using the CHROM algorithm
with robust signal validation and multi-ROI frequency consistency analysis.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple, List

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

# ── MediaPipe Tasks API ────────────────────────────────────────────────────────
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import core as mp_core
import mediapipe as mp

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

MAX_FRAMES   = 180       # Frame limit
MIN_FRAMES   = 30        # Minimum valid frames required
TARGET_FPS   = 30.0      # Fallback fps

CARDIAC_LOW  = 0.8       # ~48 BPM
CARDIAC_HIGH = 2.5       # ~150 BPM
BUTTER_ORDER = 4

# ROI Landmarks
LMKS_FOREHEAD    = np.array([10, 67, 69, 104, 108, 151, 299, 337, 338], dtype=np.int32)
LMKS_LEFT_CHEEK  = np.array([116, 117, 118, 119, 100, 142, 203, 206, 207], dtype=np.int32)
LMKS_RIGHT_CHEEK = np.array([345, 346, 347, 348, 329, 371, 423, 426, 427], dtype=np.int32)
ROI_HALF = 18            # 36x36 pixel patches

_MP_MODEL_ENV   = "MEDIAPIPE_FACE_LANDMARKER_MODEL"
_MP_MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _build_landmarker() -> Optional[mp_vision.FaceLandmarker]:
    model_path = os.environ.get(_MP_MODEL_ENV, _MP_MODEL_LOCAL)
    if not os.path.isfile(model_path):
        warnings.warn("FaceLandmarker model not found. Check path or set env var.")
        return None

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_core.base_options.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

def _landmark_centroid(landmarks, indices: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
    xs = np.array([landmarks[i].x for i in indices])
    ys = np.array([landmarks[i].y for i in indices])
    cx = int(np.clip(xs.mean() * img_w, 0, img_w - 1))
    cy = int(np.clip(ys.mean() * img_h, 0, img_h - 1))
    return cx, cy

def _extract_roi_mean(frame_rgb: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1, x2 = max(0, cx - ROI_HALF), min(w, cx + ROI_HALF)
    y1, y2 = max(0, cy - ROI_HALF), min(h, cy + ROI_HALF)
    patch = frame_rgb[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0:
        return np.zeros(3)
    return patch.reshape(-1, 3).mean(axis=0)

def _chrom_signal(rgb_series: np.ndarray) -> np.ndarray:
    mu = rgb_series.mean(axis=0)
    mu = np.where(mu < 1e-6, 1e-6, mu)
    rn = rgb_series / mu

    R, G, B = rn[:, 0], rn[:, 1], rn[:, 2]
    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    std_x, std_y = X.std(), Y.std()
    if std_x < 1e-9: return np.zeros_like(X)
    if std_y < 1e-9: return X - X.mean()

    alpha = std_x / std_y
    S = X - alpha * Y
    S -= S.mean()
    return S

def _bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    if len(signal) < 3 * BUTTER_ORDER * 2 + 1:
        return signal
    nyq = fps / 2.0
    lo = float(np.clip(CARDIAC_LOW / nyq, 1e-4, 0.99))
    hi = float(np.clip(CARDIAC_HIGH / nyq, lo + 1e-4, 0.9999))
    sos = butter(BUTTER_ORDER, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING & SCORING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _process_roi(rgb_array: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """
    Returns: (signal_strength_std, peak_hz, snr)
    """
    raw_chrom = _chrom_signal(rgb_array)
    raw_std = float(np.std(raw_chrom))
    
    filt = _bandpass(raw_chrom, fps)
    
    # FFT Analysis
    n = len(filt)
    window = np.hanning(n)
    power = np.abs(rfft(filt * window)) ** 2
    freqs = rfftfreq(n, d=1.0 / fps)
    
    cardiac_mask = (freqs >= CARDIAC_LOW) & (freqs <= CARDIAC_HIGH)
    if not cardiac_mask.any():
        return raw_std, 0.0, 0.0
        
    c_powers = power[cardiac_mask]
    peak_idx = np.argmax(c_powers)
    peak_hz = float(freqs[cardiac_mask][peak_idx])
    peak_power = c_powers[peak_idx]
    
    others_sum = c_powers.sum() - peak_power
    others_mean = others_sum / max(1, len(c_powers) - 1)
    
    snr = float(peak_power / (others_mean + 1e-12))
    
    return raw_std, peak_hz, snr

def process_video(file_path: str) -> Tuple[float, str]:
    if not os.path.isfile(file_path):
        return 0.0, "File not found."

    landmarker = _build_landmarker()
    if landmarker is None:
        return 0.35, "FaceLandmarker model missing."

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        landmarker.close()
        return 0.0, "Could not open video file."

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or TARGET_FPS
    
    rgb_buffers = [[], [], []]
    frames_read = 0
    
    try:
        while frames_read < MAX_FRAMES:
            ok, frame_bgr = cap.read()
            if not ok: break
            frames_read += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int((frames_read / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, ts)
            if not result.face_landmarks:
                continue
                
            lm = result.face_landmarks[0]
            h, w = frame_rgb.shape[:2]
            
            for i, indices in enumerate([LMKS_FOREHEAD, LMKS_LEFT_CHEEK, LMKS_RIGHT_CHEEK]):
                cx, cy = _landmark_centroid(lm, indices, w, h)
                rgb_buffers[i].append(_extract_roi_mean(frame_rgb, cx, cy))
    finally:
        cap.release()
        landmarker.close()
        
    n_valid = min(len(buf) for buf in rgb_buffers)
    if n_valid < MIN_FRAMES:
        return 0.15, "Insufficient face data detected."

    rgb_arrays = [np.stack(buf[:n_valid], axis=0) for buf in rgb_buffers]

    # Analyze each ROI
    metrics = []
    for rgb in rgb_arrays:
        metrics.append(_process_roi(rgb, fps))
        
    stds  = [m[0] for m in metrics]
    hzs   = [m[1] for m in metrics]
    snrs  = [m[2] for m in metrics]
    
    max_std = max(stds)
    mean_snr = float(np.mean(snrs))
    hz_mean = float(np.mean(hzs))
    hz_spread = max(hzs) - min(hzs)
    
    # ── 1. HARD VALIDATION ──────────────────────────────────────────────
    
    # Check 1: Is the signal physically too flat (static image)?
    # Chrominance std < 5e-5 usually perfectly defines camera hardware read noise on flat prints.
    if max_std < 5e-5:
        return 0.05, f"FAKE | Extremely low color variance ({max_std:.2e}) indicates static photo."
        
    # Check 2: Is the signal severely noisy (rapid movement or light flicker)?
    if max_std > 0.05:
        return 0.10, f"FAKE | Unstable lighting or huge artifact variation ({max_std:.2e})."
        
    # Check 3: Is there a clear rhythmic heartbeat present?
    # Mean SNR falls below 1.5 for incoherent broadband noise.
    if mean_snr < 1.5:
        return 0.15, f"FAKE | No dominant heart frequency detected (SNR: {mean_snr:.1f})."
        
    # ── 2. BALANCED SCORING ─────────────────────────────────────────────
    
    # Normalize SNR mapping bounded typically between 1.5 (noise) and 6.0+ (clean pulse)
    snr_score = np.clip((mean_snr - 1.5) / 4.5, 0.0, 1.0)
    
    # Ensure frequency consistency. Real blood flows perfectly synchronously.
    # Deviation above 0.5 Hz means noise dominates over true pulse alignment.
    if hz_spread <= 0.2:
        consistency_score = 1.0
    elif hz_spread <= 0.5:
        consistency_score = 0.5
    else:
        consistency_score = 0.0
        
    raw_score = 0.60 * snr_score + 0.40 * consistency_score
    score = float(np.clip(raw_score, 0.0, 1.0))
    
    bpm = hz_mean * 60.0
    
    if score >= 0.65:
        verdict = "REAL"
    elif score >= 0.40:
        verdict = "UNCERTAIN"
    else:
        verdict = "FAKE"
        
    explanation = (
        f"{verdict} | HR: {bpm:.1f} bpm | SNR: {mean_snr:.1f} | "
        f"Strength: {max_std:.2e} | Hz Diff: {hz_spread:.2f}Hz"
    )
    
    return score, explanation