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
# 4.  Blur-Pool — anti-aliased translation-invariant pooling
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class BlurPool2d(nn.Module):
        """
        Anti-aliased pooling (Zhang, 2019 — "Making CNNs Shift-Invariant").

        Replaces strided max-pool with a Gaussian low-pass filter applied
        *before* the stride, eliminating aliasing artefacts that cause the
        network to be sensitive to sub-pixel face shifts.

        Why this helps deepfake generalisation
        ---------------------------------------
        The pretrained EfficientNet-V2-S uses standard strided convolutions
        whose aliasing makes the predicted class flip when the face is shifted
        by 1–2 pixels — a common occurrence between real photos and screenshots.
        BlurPool removes this sensitivity, so domain shift from face alignment
        differences stops fooling the backbone.

        Parameters
        ----------
        channels  : number of input feature-map channels
        kernel_size: Gaussian kernel width (default 3; 5 for stronger smoothing)
        stride    : downsampling stride applied AFTER blurring (default 2)
        padding   : zero-padding applied before the blur kernel (default 1)

        Forward
        -------
        x   : (B, C, H, W)
        out : (B, C, H//stride, W//stride)
        """

        def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
        ) -> None:
            super().__init__()
            self.stride  = stride
            self.padding = padding

            # Build a fixed Gaussian kernel of shape (C, 1, k, k)
            if kernel_size == 3:
                filt = torch.tensor([1., 2., 1.])
            elif kernel_size == 5:
                filt = torch.tensor([1., 4., 6., 4., 1.])
            else:
                raise ValueError(f"BlurPool2d: kernel_size must be 3 or 5, got {kernel_size}")

            filt = filt[:, None] * filt[None, :]          # outer product → (k, k)
            filt = filt / filt.sum()                       # normalise
            filt = filt[None, None, :, :].repeat(channels, 1, 1, 1)  # (C, 1, k, k)

            # Register as a non-trainable buffer so it moves with .to(device)
            self.register_buffer("blur_kernel", filt)
            self.channels = channels

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.nn.functional.conv2d(
                x,
                self.blur_kernel,
                stride=self.stride,
                padding=self.padding,
                groups=self.channels,
            )

else:  # pragma: no cover
    class BlurPool2d:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise ImportError("pip install torch")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Stream A — EfficientNet-V2-S Spatial Backbone (with BlurPool)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE and _TIMM_AVAILABLE:

    class SpatialStream(nn.Module):
        """
        Stream A — Spatial feature extractor.

        Architecture
        ------------
        EfficientNet-V2-S (timm), pretrained on ImageNet-21k → ImageNet-1k
          → BlurPool2d injected at the stem stride-2 conv  (translation invariance)
          → Global average pool  →  1280-d feature vector  (no classifier head)

        BlurPool Modification
        ---------------------
        The EfficientNet stem contains a single Conv2d with stride=2.  We wrap
        it with BlurPool2d so the downsampling step goes through a low-pass
        filter first.  This makes the network insensitive to 1–2 pixel face
        shifts between domains (real camera vs screenshot vs social media re-encode).

        Partial-Unfreeze API
        --------------------
        By default the entire backbone is trainable. Call ``partial_unfreeze(n=2)``
        to freeze all EfficientNet blocks *except* the last ``n``, then fine-tune
        the exposed blocks with a very low LR (1e-5) for 3 epochs.  This lets the
        high-level semantic layers adapt to diverse deepfake distributions without
        disturbing the low-level texture detectors.

        Parameters
        ----------
        pretrained  : load ImageNet weights from timm hub  (default True)
        freeze_bn   : freeze all BN statistics (useful for fine-tune)  (default False)
        inject_blurpool : patch the stem with BlurPool2d  (default True)

        Forward
        -------
        x   : (B, 3, H, W)  float32 tensor, ImageNet-normalised
        out : (B, 1280)      float32 feature vector
        """

        OUT_DIM = 1280   # EfficientNet-V2-S penultimate feature dimension

        def __init__(
            self,
            pretrained: bool = True,
            freeze_bn: bool = False,
            inject_blurpool: bool = True,
        ) -> None:
            super().__init__()

            self.backbone = timm.create_model(
                "tf_efficientnetv2_s",
                pretrained=pretrained,
                num_classes=0,       # remove the classification head
                global_pool="avg",   # return (B, 1280) after pooling
            )

            # ── Inject BlurPool into the stem strided conv ────────────────────
            if inject_blurpool:
                self._inject_blurpool()

            # ── Optionally freeze BatchNorm stats ─────────────────────────────
            if freeze_bn:
                for m in self.backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False

            logger.info(
                "SpatialStream: EfficientNet-V2-S loaded "
                "(pretrained=%s, out_dim=%d, blurpool=%s)",
                pretrained, self.OUT_DIM, inject_blurpool,
            )

        # ── BlurPool injection ────────────────────────────────────────────────
        def _inject_blurpool(self) -> None:
            """
            Replace the first stride-2 Conv2d in the EfficientNet stem with
            a (Conv2d stride-1) → BlurPool2d(stride=2) sequence.

            EfficientNet-V2-S stem layout (timm default):
              conv_stem: Conv2d(3, 24, 3, stride=2, padding=1)
              → We split the stride: conv keeps stride=1, BlurPool does stride=2.
            """
            stem_conv = self.backbone.conv_stem          # Conv2d(3, 24, 3, stride=2)
            out_channels = stem_conv.out_channels        # 24

            # New conv with stride=1 (anti-alias before downsampling)
            new_conv = nn.Conv2d(
                stem_conv.in_channels,
                out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=1,                                # stride moved to BlurPool
                padding=stem_conv.padding,
                bias=False,
            )
            # Copy original weights so we don't break pretrained features
            with torch.no_grad():
                new_conv.weight.copy_(stem_conv.weight)

            # Replace stem with: conv(stride=1) → BlurPool(stride=2)
            self.backbone.conv_stem = nn.Sequential(
                new_conv,
                BlurPool2d(channels=out_channels, kernel_size=3, stride=2, padding=1),
            )
            logger.info("SpatialStream: BlurPool2d injected into EfficientNet stem.")

        # ── Partial unfreeze for domain fine-tuning ───────────────────────────
        def partial_unfreeze(self, n_blocks: int = 2) -> None:
            """
            Freeze all EfficientNet-V2-S blocks *except* the last ``n_blocks``.

            Use this for the 3-epoch domain generalisation fine-tune phase:
              stream_a.partial_unfreeze(n_blocks=2)
              # then set backbone LR to 1e-5 in the optimiser

            EfficientNet-V2-S has 6 MBConv/FusedMBConv stages (blocks 0–5).
            Freezing 0–3 and releasing 4–5 lets the high-level feature detectors
            adapt while keeping low-level texture knowledge intact.

            Parameters
            ----------
            n_blocks : number of EfficientNet blocks (stages) to unfreeze from the end.
            """
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Collect the named block groups (timm names them blocks.0 … blocks.5)
            block_groups = []
            for name, module in self.backbone.named_children():
                if name.startswith("block"):
                    block_groups.append((name, module))

            # Unfreeze the last n_blocks groups
            for name, module in block_groups[-n_blocks:]:
                for param in module.parameters():
                    param.requires_grad = True
                logger.info("SpatialStream.partial_unfreeze: unfrozen → %s", name)

            # Always keep the head (conv_head + bn + classifier) trainable
            for part_name in ("conv_head", "bn2", "classifier"):
                part = getattr(self.backbone, part_name, None)
                if part is not None:
                    for param in part.parameters():
                        param.requires_grad = True

            n_trainable = sum(p.requires_grad for p in self.backbone.parameters())
            n_total     = sum(1 for _ in self.backbone.parameters())
            logger.info(
                "SpatialStream.partial_unfreeze: %d / %d params trainable "
                "(last %d blocks + head)",
                n_trainable, n_total, n_blocks,
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