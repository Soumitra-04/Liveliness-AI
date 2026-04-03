"""
app/ai_engine/spatial.py
========================
Liveliness-AI  |  Stream A — Spatial Backbone (EfficientNet-V2-S)

Public API (inference helpers — unchanged)
------------------------------------------
  spatial_inconsistency_score(file_path) → float

Training components (NEW)
--------------------------
  DegradationTransform   — on-the-fly JPEG / Blur augmentation (20 % prob)
  DeepFakeDataset        — CSV-backed PyTorch Dataset
  SpatialStream          — EfficientNet-V2-S backbone; returns 1280-d features

Architecture notes — Stream A
------------------------------
  EfficientNet-V2-S (timm) pretrained on ImageNet-1k
    ↓  global average pool
    ↓  1280-d feature vector
  → concatenated in fusion.py with frequency features from noise.py
"""

from __future__ import annotations

import io
import logging
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy-weight imports — graceful degradation when torch missing ───

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:                  # pragma: no cover
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed — Dataset / DegradationTransform unavailable.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    import torchvision
    _TORCH_AVAILABLE = True
except ImportError:                  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — SpatialStream / Dataset unavailable.")

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:                  # pragma: no cover
    _TIMM_AVAILABLE = False
    logger.warning("timm not installed — SpatialStream unavailable. pip install timm")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:                  # pragma: no cover
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed — DeepFakeDataset unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Original spatial analysis helper  (API preserved — used by image_ela.py)
# ══════════════════════════════════════════════════════════════════════════════

def spatial_inconsistency_score(file_path: str) -> float:
    """
    Compute a spatial-inconsistency score for a single image using Canny edges.

    Returns
    -------
    float  in [0.0, 1.0].   Higher = more spatial anomaly.
    """
    img = cv2.imread(file_path)
    if img is None:
        logger.warning("spatial_inconsistency_score: cannot read '%s'", file_path)
        return 0.0
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    score = np.std(edges) / 255.0
    return float(min(score, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  On-the-fly degradation transform
# ══════════════════════════════════════════════════════════════════════════════

class DegradationTransform:
    """
    Randomly degrades a PIL Image with JPEG artefacts, Gaussian blur,
    or a screenshot simulation (PNG round-trip via numpy).

    Applied to a random ``degradation_prob`` fraction of training samples
    (default 20 %).  Designed to survive blur so that checkerboard frequency
    artefacts become the discriminating signal for Stream B.

    The screenshot mode ('png_roundtrip') simulates the OS compositor
    re-rendering pipeline: PIL → numpy uint8 → PIL.  This destroys the
    microscopic GAN phase artefacts the same way a real screenshot does,
    forcing the model to learn screenshot-robust spatial features.

    Parameters
    ----------
    degradation_prob      : probability a sample is degraded  (default 0.20)
    jpeg_quality_range    : (min_q, max_q) for JPEG round-trip (default 20-75)
    blur_radius_range     : (min_σ, max_σ) for Gaussian blur  (default 0.5-2.5)
    screenshot_prob       : fraction of degraded samples that use screenshot
                           simulation (default 0.15 — i.e. 15% of 20%)
    """

    def __init__(
        self,
        degradation_prob: float = 0.20,
        jpeg_quality_range: Tuple[int, int] = (20, 75),
        blur_radius_range: Tuple[float, float] = (0.5, 2.5),
        screenshot_prob: float = 0.15,
    ) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow required: pip install Pillow")
        self.degradation_prob   = degradation_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_radius_range  = blur_radius_range
        self.screenshot_prob    = screenshot_prob

    def __call__(self, img: "PILImage.Image") -> "PILImage.Image":
        if random.random() >= self.degradation_prob:
            return img
        # Choose degradation mode
        r = random.random()
        if r < self.screenshot_prob:
            return self._apply_screenshot(img)
        mode = random.choice(["jpeg", "blur", "both"])
        if mode in ("jpeg", "both"):
            img = self._apply_jpeg(img)
        if mode in ("blur", "both"):
            img = self._apply_blur(img)
        return img

    def _apply_jpeg(self, img: "PILImage.Image") -> "PILImage.Image":
        quality = random.randint(*self.jpeg_quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return PILImage.open(buf).copy()

    def _apply_blur(self, img: "PILImage.Image") -> "PILImage.Image":
        from PIL import ImageFilter
        sigma = random.uniform(*self.blur_radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _apply_screenshot(self, img: "PILImage.Image") -> "PILImage.Image":
        """
        Simulate an OS screenshot by round-tripping through numpy uint8.
        This destroys microscopic GAN phase artefacts in the high-frequency
        domain — the same degradation path as a real Windows/macOS screenshot.
        Optional: adds a subtle PNG re-compression step via BytesIO.
        """
        # PIL → numpy uint8 (mimics screen buffer quantisation)
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        # numpy → PIL (mimics screenshot capture)
        reconstructed = PILImage.fromarray(arr, mode="RGB")
        # Optional PNG buffer round-trip (mimics OS PNG compression)
        buf = io.BytesIO()
        reconstructed.save(buf, format="PNG", compress_level=1)
        buf.seek(0)
        return PILImage.open(buf).copy()

    def __repr__(self) -> str:
        return (f"DegradationTransform(prob={self.degradation_prob}, "
                f"jpeg={self.jpeg_quality_range}, blur_σ={self.blur_radius_range}, "
                f"screenshot_prob={self.screenshot_prob})")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CSV-backed PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE and _PIL_AVAILABLE and _PANDAS_AVAILABLE:

    class DeepFakeDataset(Dataset):
        """
        PyTorch Dataset backed by train/valid/test CSV files.

        CSV columns expected:  ,original_path,id,label,label_str,path
          label : 1 = real, 0 = fake
          path  : e.g.  "train/real/31355.jpg"  (relative to ``base_dir``)

        Parameters
        ----------
        csv_path   : path to split CSV
        base_dir   : root containing train/, valid/, test/ image folders
        split      : "train" | "valid" | "test"  (for logging only)
        transform  : torchvision transform applied after PIL load
        augment    : prepend DegradationTransform when split=="train"
        degradation_kwargs : forwarded to DegradationTransform

        __getitem__ returns (image, label_tensor, abs_path_str)
        """

        def __init__(
            self,
            csv_path: str | Path,
            base_dir: str | Path,
            split: str = "train",
            transform: Optional[Callable] = None,
            augment: bool = False,
            degradation_kwargs: Optional[dict] = None,
        ) -> None:
            super().__init__()
            self.split    = split
            self.base_dir = Path(base_dir).resolve()
            self.transform = transform

            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            for col in ("path", "label"):
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' missing from {csv_path.name}")
            self._df = df.reset_index(drop=True)

            self._degradation: Optional[DegradationTransform] = None
            if augment and split == "train":
                self._degradation = DegradationTransform(**(degradation_kwargs or {}))
                logger.info("DeepFakeDataset[train] augmentation: %s", self._degradation)

            counts = self.class_counts()
            logger.info(
                "DeepFakeDataset[%s]  total=%d  real=%d  fake=%d",
                split, counts["total"], counts["real"], counts["fake"],
            )

        def __len__(self) -> int:
            return len(self._df)

        def __getitem__(self, idx: int):
            row      = self._df.iloc[idx]
            abs_path = self.base_dir / str(row["path"])

            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {abs_path}\n"
                    "  Run dataset_setup.py to extract the archive first."
                )

            img = PILImage.open(abs_path).convert("RGB")

            if self._degradation is not None:
                img = self._degradation(img)

            if self.transform is not None:
                img = self.transform(img)

            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img, label, str(abs_path)

        def class_counts(self) -> dict:
            c = self._df["label"].value_counts().to_dict()
            return {"real": c.get(1, 0), "fake": c.get(0, 0), "total": len(self._df)}

        def __repr__(self) -> str:
            c = self.class_counts()
            return (f"DeepFakeDataset(split='{self.split}', total={c['total']}, "
                    f"real={c['real']}, fake={c['fake']}, base='{self.base_dir}')")

else:                                # pragma: no cover
    class DeepFakeDataset:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            pkgs = [p for p, ok in [("torch", _TORCH_AVAILABLE),
                                     ("Pillow", _PIL_AVAILABLE),
                                     ("pandas", _PANDAS_AVAILABLE)] if not ok]
            raise ImportError(f"Missing: {pkgs}.  pip install {' '.join(pkgs)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Stream A — EfficientNet-V2-S Spatial Backbone
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE and _TIMM_AVAILABLE:

    class SpatialStream(nn.Module):
        """
        Stream A — Spatial feature extractor.

        Architecture
        ------------
        EfficientNet-V2-S (timm), pretrained on ImageNet-1k
          → Global average pool  →  1280-d feature vector  (no classifier head)

        The 1280-d output is concatenated with the frequency features produced
        by FrequencyStream (noise.py) inside the DeepFakeFusionHead (fusion.py).

        Parameters
        ----------
        pretrained  : load ImageNet-1k weights from timm hub  (default True)
        freeze_bn   : freeze BatchNorm statistics for fine-tuning  (default False)

        Forward
        -------
        x   : (B, 3, H, W)  float32 tensor, ImageNet-normalised
        out : (B, 1280)      float32 feature vector
        """

        OUT_DIM = 1280   # EfficientNet-V2-S penultimate feature dimension

        def __init__(self, pretrained: bool = True, freeze_bn: bool = False) -> None:
            super().__init__()

            self.backbone = timm.create_model(
                "tf_efficientnetv2_s",
                pretrained=pretrained,
                num_classes=0,          # remove the classification head
                global_pool="avg",      # return (B, 1280) after pooling
            )

            if freeze_bn:
                for m in self.backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False

            logger.info(
                "SpatialStream: EfficientNet-V2-S loaded (pretrained=%s, out_dim=%d)",
                pretrained, self.OUT_DIM,
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.backbone(x)          # (B, 1280)

else:                                        # pragma: no cover
    class SpatialStream:  # type: ignore[no-redef]
        """Stub when torch / timm are not installed."""
        OUT_DIM = 1280
        def __init__(self, *_, **__):
            raise ImportError("pip install torch timm")
        def forward(self, x):
            raise ImportError("pip install torch timm")