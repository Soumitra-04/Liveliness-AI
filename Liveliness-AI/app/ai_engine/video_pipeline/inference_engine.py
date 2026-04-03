"""
inference_engine.py — Batch-capable EfficientNet-B0 inference engine.

Model output
-------------
  class 0 = REAL  (prob_real)
  class 1 = FAKE  (prob_fake = 1 - prob_real)

All public functions return FAKE-probability ∈ [0, 1].
  0.0 → certainly real
  1.0 → certainly fake

Batch inference
---------------
Faces are stacked into batches of BATCH_SIZE before forwarding through
the network.  A single run_inference() call still works correctly.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import cv2

# ── Singletons ─────────────────────────────────────────────────────────────────
_detector: Optional["DeepfakeDetector"] = None  # type: ignore[name-defined]
_torch   : Optional[object]             = None

BATCH_SIZE = 16  # process this many faces per forward pass

# ImageNet normalisation (must match infer.py)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_detector():
    global _detector
    if _detector is None:
        from app.ai_engine.deepfake_model.infer import DeepfakeDetector
        print("[inference_engine] Loading deepfake model (first request)…")
        _detector = DeepfakeDetector()
        print("[inference_engine] Model loaded OK.")
    return _detector


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def _face_to_tensor(face_bgr: np.ndarray) -> "torch.Tensor":
    """Convert one BGR face to a (3, 224, 224) float32 tensor (no batch dim)."""
    torch = _get_torch()
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))   # (3, 224, 224)


def run_inference(face_bgr: np.ndarray) -> float:
    """
    Run inference on a single face.

    Parameters
    ----------
    face_bgr : np.ndarray  BGR image, any size.

    Returns
    -------
    float  Probability of REAL in [0, 1].
    """
    return run_batch_inference([face_bgr])[0]


def run_batch_inference(faces_bgr: List[np.ndarray]) -> List[float]:
    """
    Run batch inference on a list of face crops.

    Processes faces in chunks of BATCH_SIZE to avoid OOM on CPU.

    Parameters
    ----------
    faces_bgr : List[np.ndarray]   BGR face images, any size.

    Returns
    -------
    List[float]  Real-probability score ∈ [0, 1] per face (same order).
    """
    torch    = _get_torch()
    detector = _get_detector()
    model    = detector._model
    device   = detector._device

    scores: List[float] = []

    for chunk_start in range(0, len(faces_bgr), BATCH_SIZE):
        chunk = faces_bgr[chunk_start: chunk_start + BATCH_SIZE]

        tensors = []
        for face in chunk:
            if face is None or face.size == 0:
                tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))
            else:
                tensors.append(_face_to_tensor(face))

        batch = torch.stack(tensors).to(device)       # (N, 3, 224, 224)

        with torch.no_grad():
            logits = model(batch)                      # (N, 2)
            probs  = torch.softmax(logits, dim=1)      # (N, 2)
            # class 0 = REAL → flip to fake-probability
            fake_probs = probs[:, 1].cpu().tolist()    # class 1 = FAKE

        scores.extend(fake_probs)

    return scores
