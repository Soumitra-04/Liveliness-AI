"""
face_aligner.py — Align a face crop to a canonical pose.

Strategy (no dlib / no external landmark model required):
  1. If an eye-region approximation is recoverable from the raw box → rotate.
  2. Fallback: square-pad and centre-crop to 224×224.

The primary goal is a stable, well-framed 224×224 face crop fed to the model.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional


def align_face(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    output_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Return an aligned BGR face crop of shape (output_size, output_size, 3).

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    box : Tuple[int, int, int, int]
        (x1, y1, x2, y2) bounding box from face_detector.
    output_size : int
        Target square size (default 224 to match EfficientNet input).

    Returns
    -------
    Optional[np.ndarray]
        Aligned face crop, or None on failure.
    """
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]

    # Add 20 % padding around the detected box for context
    bw    = x2 - x1
    bh    = y2 - y1
    pad_x = int(bw * 0.20)
    pad_y = int(bh * 0.20)
    x1p   = max(0, x1 - pad_x)
    y1p   = max(0, y1 - pad_y)
    x2p   = min(W, x2 + pad_x)
    y2p   = min(H, y2 + pad_y)

    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None

    # Make the crop square (take the longer side → pad shorter side)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    canvas[y_off : y_off + ch, x_off : x_off + cw] = crop

    # Resize to model input size
    aligned = cv2.resize(canvas, (output_size, output_size))
    return aligned
