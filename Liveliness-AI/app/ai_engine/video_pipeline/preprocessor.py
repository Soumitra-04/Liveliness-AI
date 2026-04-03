"""
preprocessor.py — Convert an aligned BGR face crop to a normalised PyTorch tensor.

ImageNet normalisation matches the EfficientNet-B0 training regime.
"""

from __future__ import annotations

import cv2
import numpy as np

# ImageNet statistics
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_face(face_bgr: np.ndarray) -> "torch.Tensor":
    """
    Convert a 224×224 BGR crop to a (1, 3, 224, 224) float32 tensor.

    Parameters
    ----------
    face_bgr : np.ndarray
        BGR image of any size — will be resized to 224×224 internally.

    Returns
    -------
    torch.Tensor
        Normalised tensor ready for EfficientNet-B0 inference.
    """
    import torch

    # BGR → RGB, resize to 224×224, normalise to [0, 1]
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0

    # ImageNet normalisation
    rgb = (rgb - _MEAN) / _STD                 # (H, W, 3)

    # HWC → CHW, add batch dimension
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,224,224)
    return tensor
