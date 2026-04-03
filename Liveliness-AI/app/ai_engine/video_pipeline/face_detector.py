"""
face_detector.py — Detect the largest face in a frame using OpenCV DNN SSD.

The ResNet-SSD model (~10 MB) is downloaded once and cached locally.
No native compilation required — pure OpenCV DNN.
"""

from __future__ import annotations

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional, Tuple

# ── Model weights ──────────────────────────────────────────────────────────────
_CACHE_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
_PROTO_URL    = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                 "samples/dnn/face_detector/deploy.prototxt")
_WEIGHTS_URL  = ("https://github.com/opencv/opencv_3rdparty/raw/"
                 "dnn_samples_face_detector_20170830/"
                 "res10_300x300_ssd_iter_140000.caffemodel")
_PROTO_PATH   = os.path.join(_CACHE_DIR, "deploy.prototxt")
_WEIGHTS_PATH = os.path.join(_CACHE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_MIN_CONFIDENCE = 0.60
_MIN_FACE_PX    = 40     # minimum face side length in pixels


def _download_models() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not os.path.isfile(_PROTO_PATH):
        print("[face_detector] Downloading SSD config …")
        urllib.request.urlretrieve(_PROTO_URL, _PROTO_PATH)
    if not os.path.isfile(_WEIGHTS_PATH):
        print("[face_detector] Downloading SSD weights (~10 MB) …")
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)


# ── Singleton face network ─────────────────────────────────────────────────────
_net: Optional[cv2.dnn.Net] = None


def _get_net() -> cv2.dnn.Net:
    global _net
    if _net is None:
        _download_models()
        _net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _WEIGHTS_PATH)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return _net


def detect_largest_face(
    frame: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest (highest confidence) face in `frame`.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        (x1, y1, x2, y2) bounding box in pixels, or None if no face found.
    """
    net = _get_net()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0), swapRB=False
    )
    net.setInput(blob)
    dets = net.forward()

    best_conf, best_box = 0.0, None
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < _MIN_CONFIDENCE:
            continue
        x1 = int(np.clip(dets[0, 0, i, 3] * w, 0, w))
        y1 = int(np.clip(dets[0, 0, i, 4] * h, 0, h))
        x2 = int(np.clip(dets[0, 0, i, 5] * w, 0, w))
        y2 = int(np.clip(dets[0, 0, i, 6] * h, 0, h))
        if (x2 - x1) < _MIN_FACE_PX or (y2 - y1) < _MIN_FACE_PX:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box  = (x1, y1, x2, y2)

    return best_box
