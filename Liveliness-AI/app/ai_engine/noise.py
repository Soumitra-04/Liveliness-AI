"""
app/ai_engine/noise.py
======================
Liveliness-AI  |  Stream B — Frequency / Phase-Spectrum Branch

Public API (inference helper — unchanged)
-----------------------------------------
  noise_score(file_path) → float        (Laplacian variance, original)

Training component (NEW)
-------------------------
  FrequencyStream  — FFT phase-spectrum CNN branch

Architecture notes — Stream B
------------------------------
  Input image (B, 3, H, W)
    ↓  per-channel 2-D FFT  →  phase spectrum  →  stack to (B, 3, H, W)
    ↓  lightweight CNN  (3×Conv-BN-ReLU blocks, stride-2 downsampling)
    ↓  Global average pool
    ↓  512-d feature vector
  → concatenated in fusion.py with 1280-d spatial features from spatial.py

Why phase, not magnitude?
--------------------------
  GAN-generated images carry spatially periodic 'checkerboard' artefacts in
  the phase spectrum that persist even after Gaussian blur (blur only dampens
  high-frequency *magnitude*; it does NOT destroy phase coherence of the
  periodic pattern).  Extracting the phase thus gives the network a signal
  that survives the DegradationTransform applied during training.

  Implementation detail:
    phase = torch.angle(torch.fft.fft2(x_grey))
    phase is shifted to centre DC  (torch.fft.fftshift)
    values are clamped to [-π, π] and normalised to [0, 1]
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional ML imports ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:              # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — FrequencyStream unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Original noise helper (API preserved — used by image_ela.py)
# ══════════════════════════════════════════════════════════════════════════════

def noise_score(file_path: str) -> float:
    """
    Estimate image noise / sharpness via Laplacian variance.

    Returns
    -------
    float  in [0.0, 1.0].  Higher = noisier / less smooth.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("noise_score: cannot read '%s'", file_path)
        return 0.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    score     = np.var(laplacian) / 1000.0
    return float(min(score, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Phase-spectrum extraction (differentiable, works in-graph)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class _PhaseExtractor(nn.Module):
        """
        Converts an RGB image tensor to a 3-channel phase-spectrum image.

        The transform is:
          1. Convert each colour channel to float
          2. 2-D FFT  →  complex64 tensor
          3. fftshift to centre the DC component
          4. Take element-wise angle (torch.angle)  →  real in [-π, π]
          5. Normalise to [0, 1]  (add π, divide by 2π)

        This module has NO learnable parameters — it is a deterministic
        pre-processing step applied before the CNN encoder.

        Forward
        -------
        x   : (B, 3, H, W)  float32   (any ImageNet-normalised input works)
        out : (B, 3, H, W)  float32   in [0, 1]
        """

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x : (B, C, H, W)
            # FFT per channel
            xc    = x.to(torch.float32)
            fft2  = torch.fft.fft2(xc)                    # complex (B, C, H, W)
            fft2s = torch.fft.fftshift(fft2, dim=(-2, -1)) # shift DC to centre
            phase = torch.angle(fft2s)                     # real, [-π, π]
            # Normalise to [0, 1]
            phase = (phase + math.pi) / (2 * math.pi)
            return phase.clamp(0.0, 1.0)


    class FrequencyStream(nn.Module):
        """
        Stream B — Frequency / Phase-Spectrum feature extractor.

        Architecture
        ------------
        PhaseExtractor (no params)          →  (B, 3, H, W) phase map
          ↓ Conv 3→32, stride 2, BN, ReLU   →  (B, 32, H/2, W/2)
          ↓ Conv 32→64, stride 2, BN, ReLU  →  (B, 64, H/4, W/4)
          ↓ Conv 64→128, stride 2, BN, ReLU →  (B, 128, H/8, W/8)
          ↓ Conv 128→256, stride 2, BN, ReLU→  (B, 256, H/16, W/16)
          ↓ Global average pool              →  (B, 256)
          ↓ Linear 256 → 512, ReLU          →  (B, 512)

        OUT_DIM = 512

        The 512-d vector is concatenated with SpatialStream's 1280-d output
        inside DeepFakeFusionHead (fusion.py) → total concat dim = 1792.

        Parameters
        ----------
        None — the CNN is trained from scratch (phase features are dataset-
        specific and ImageNet weights bring no benefit here).

        Forward
        -------
        x   : (B, 3, H, W)  float32
        out : (B, 512)       float32  frequency feature vector
        """

        OUT_DIM = 512

        def __init__(self) -> None:
            super().__init__()

            self.phase_extractor = _PhaseExtractor()

            def _block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            self.encoder = nn.Sequential(
                _block(3,   32,  stride=2),   # H/2
                _block(32,  64,  stride=2),   # H/4
                _block(64,  128, stride=2),   # H/8
                _block(128, 256, stride=2),   # H/16
            )
            self.pool      = nn.AdaptiveAvgPool2d(1)
            self.projector = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, self.OUT_DIM),
                nn.ReLU(inplace=True),
            )

            # Initialise CNN weights (phase features ≠ ImageNet features)
            self._init_weights()

            logger.info("FrequencyStream: phase-spectrum CNN  out_dim=%d", self.OUT_DIM)

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            phase   = self.phase_extractor(x)   # (B, 3, H, W)
            feats   = self.encoder(phase)        # (B, 256, H/16, W/16)
            pooled  = self.pool(feats)           # (B, 256, 1, 1)
            out     = self.projector(pooled)     # (B, 512)
            return out

else:                                            # pragma: no cover
    class FrequencyStream:  # type: ignore[no-redef]
        OUT_DIM = 512
        def __init__(self, *_, **__):
            raise ImportError("pip install torch")
        def forward(self, x):
            raise ImportError("pip install torch")