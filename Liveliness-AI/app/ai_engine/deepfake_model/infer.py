"""
infer.py — DeepfakeDetector inference wrapper.

Architecture:
  EfficientNet-B0 backbone (torchvision, ImageNet pretrained)
  Custom classifier head:
    Dropout(0.4) → Linear(1280 → 2)
  Weights: best_model-v3.pt  (2-class: 0=REAL, 1=FAKE)

Usage:
    from app.ai_engine.deepfake_model.infer import DeepfakeDetector

    detector = DeepfakeDetector()          # loads weights once
    prob_real = detector.predict(face_bgr) # numpy BGR image → float [0,1]
"""

from __future__ import annotations

import os
import warnings
from typing import Union

import cv2
import numpy as np

# Weight file path (place best_model-v3.pt in the weights/ sub-directory)
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best_model-v3.pt")

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(face_bgr: np.ndarray) -> "torch.Tensor":
    """Resize, normalise and convert a BGR numpy image to a (1,3,224,224) tensor."""
    import torch
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD            # ImageNet normalisation
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return tensor


def _build_model() -> "torch.nn.Module":
    """Construct EfficientNet-B0 with the custom classification head."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    import torch.nn as nn

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2),
    )
    return model


class DeepfakeDetector:
    """
    Singleton-friendly wrapper around the EfficientNet-B0 deepfake classifier.

    Parameters
    ----------
    weights_path : str, optional
        Override the default weights path.
    device : str, optional
        'cpu' or 'cuda'.  Defaults to 'cpu' for portability.
    """

    def __init__(self, weights_path: str = _WEIGHTS_PATH, device: str = "cpu"):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required. Install with:\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu"
            )

        self._device = torch.device(device)
        self._model  = _build_model().to(self._device)

        if not os.path.isfile(weights_path):
            warnings.warn(
                f"[DeepfakeDetector] Weights not found at '{weights_path}'. "
                "The model will return uninitialized predictions. "
                "Download best_model-v3.pt from:\n"
                "  https://github.com/TRahulsingh/DeepfakeDetector/tree/main/models\n"
                "and place it in:\n"
                f"  {os.path.dirname(weights_path)}",
                stacklevel=2,
            )
        else:
            import torch
            state = torch.load(weights_path, map_location=self._device)
            # Robustly unwrap lightning checkpoints or raw state dicts
            if isinstance(state, dict):
                if "state_dict" in state:
                    # PyTorch-Lightning checkpoint
                    sd = {k.replace("model.", "", 1): v
                          for k, v in state["state_dict"].items()
                          if k.startswith("model.")}
                else:
                    sd = state
            else:
                sd = state
            self._model.load_state_dict(sd, strict=False)

        self._model.eval()

    def predict(self, face_img: np.ndarray) -> float:
        """
        Classify a single face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or RGB image (any size — will be resized to 224×224 internally).

        Returns
        -------
        float
            Probability of being REAL in [0.0, 1.0].
            High value → real. Low value → deepfake.
        """
        if face_img is None or face_img.size == 0:
            return 0.5  # uncertain

        import torch
        tensor = _preprocess(face_img).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)          # (1, 2)
            probs  = torch.softmax(logits, dim=1) # (1, 2)
            prob_real = float(probs[0, 0].item()) # class 0 = REAL

        return prob_real
