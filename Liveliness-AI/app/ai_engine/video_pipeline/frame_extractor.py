"""
frame_extractor.py — Extract up to MAX_FRAMES frames uniformly from a video.

Uniform sampling ensures we cover beginning, middle, and end of the clip
regardless of duration, and cap at 50 frames for performance.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List

MAX_FRAMES = 50   # hard cap — enough signal, bounded latency


def extract_frames(video_path: str, target_fps: float = 2.0) -> List[np.ndarray]:
    """
    Extract up to MAX_FRAMES frames from `video_path`, sampled uniformly.

    Strategy
    --------
    1. Compute how many frames to skip based on `target_fps`.
    2. If total candidate frames > MAX_FRAMES, further sub-sample uniformly.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    target_fps : float
        Approximate sampling rate (default 2 fps).

    Returns
    -------
    List[np.ndarray]
        BGR frames as numpy arrays.  Empty list if video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step        = max(1, int(round(video_fps / target_fps)))

    # Indices of candidate frames (every `step` frames)
    candidate_indices = list(range(0, max(total_count, 1), step))

    # Further sub-sample if more than MAX_FRAMES
    if len(candidate_indices) > MAX_FRAMES:
        sub_step       = len(candidate_indices) / MAX_FRAMES
        candidate_indices = [
            candidate_indices[int(i * sub_step)]
            for i in range(MAX_FRAMES)
        ]

    candidate_set = set(candidate_indices)
    frames: List[np.ndarray] = []
    frame_idx = 0

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in candidate_set:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames
