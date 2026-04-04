"""
app/ai_engine/fusion.py
=======================
Liveliness-AI  |  Stream C — Dual-Stream Fusion Head + PyTorch Lightning Module

Public API (inference helper — unchanged)
-----------------------------------------
  combine_results(image_result, video_result, audio_result) → FusionOutput

Training components (NEW)
--------------------------
  DeepFakeFusionHead   — concatenates spatial + frequency streams + Dropout(0.5)
  DeepFakeV1Module     — PyTorch Lightning LightningModule wrapping all three streams

Architecture — complete pipeline
---------------------------------
  Input image (B, 3, 224, 224)
       │
       ├─► SpatialStream (EfficientNet-V2-S pretrained)  →  (B, 1280)  ──┐
       │                                                                   │
       └─► FrequencyStream (Phase-Spectrum CNN)         →  (B,  512)  ──┤
                                                                          │
                                                             concat  →  (B, 1792)
                                                             BN      →  (B, 1792)
                                                             Dropout(0.5)
                                                             Linear 1792→512, ReLU
                                                             Dropout(0.3)
                                                             Linear  512→2
                                                             ↓
                                                          logits / prediction

Training configuration
----------------------
  Optimizer  : AdamW  (lr=1e-4, weight_decay=1e-2)
  LR schedule: CosineAnnealingLR  (T_max = num_epochs)
  Loss       : CrossEntropyLoss   (label-smoothing=0.05)
  Precision  : 16-bit mixed  (configured externally via Trainer(precision=16))
  Epochs     : 5–8  (user-specified)
  Checkpoint : best val_acc  → models/ml_model/best_deepfake_v1.pth
"""

from __future__ import annotations

import logging
from typing import Tuple, TypedDict
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

# ── Optional ML imports ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:              # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — DeepFakeFusionHead / V1Module unavailable.")

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    _PL_AVAILABLE = True
except ImportError:              # pragma: no cover
    _PL_AVAILABLE = False
    logger.warning("PyTorch Lightning not installed — pip install pytorch-lightning")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Original runtime fusion helper  (API preserved — used by main.py routes)
# ══════════════════════════════════════════════════════════════════════════════

ModalityResult = Tuple[float, str]
OptionalModalityResult = Optional[ModalityResult]


class FusionOutput(TypedDict):
    authenticity_score:  float
    risk_classification: str
    flags:               list[str]


# Image carries the strongest signal (EfficientNet-V2-S backbone).
# Heuristic video/audio streams are down-weighted to reduce false positives.
_WEIGHTS: dict[str, float] = {"image": 0.60, "video": 0.20, "audio": 0.20}

_RISK_BANDS: list[tuple[float, str]] = [
    (70.0, "LOW"),
    (40.0, "MEDIUM"),
    (0.0,  "HIGH"),
]


def combine_results(
    image_result: Optional[ModalityResult] = None,
    video_result: Optional[ModalityResult] = None,
    audio_result: Optional[ModalityResult] = None,
) -> FusionOutput:
    """
    Merge per-modality (score, explanation) pairs into a single risk verdict.
    Score range: [0.0, 1.0]  — 0 = fake, 1 = real.

    Dynamic Normalization
    ---------------------
    Only modalities that are explicitly provided (not None) contribute to the
    final score.  Their weights are re-normalised so they always sum to 1.0.

    Example — image-only upload with score=0.98:
      active_weight  = _WEIGHTS["image"] = 0.60
      raw_score      = (0.98 * 0.60) / 0.60 = 0.98
      authenticity_score = 98.0  (not 39.2 as before)
    """
    all_results: dict[str, Optional[ModalityResult]] = {
        "image": image_result,
        "video": video_result,
        "audio": audio_result,
    }

    # ── Validate scores for present modalities ───────────────────────────────
    active: dict[str, ModalityResult] = {}
    for name, result in all_results.items():
        if result is None:
            continue
        score, _ = result
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"Score for '{name}' is {score!r} — must be in [0.0, 1.0]."
            )
        active[name] = result

    if not active:
        raise ValueError("At least one modality result must be provided.")

    # ── Dynamic weighted average (normalised over active modalities only) ────
    total_weight = sum(_WEIGHTS[n] for n in active)
    raw_score    = sum(_WEIGHTS[n] * s for n, (s, _) in active.items()) / total_weight

    authenticity_score  = max(0.0, min(100.0, round(float(raw_score * 100), 1)))
    risk_classification = _classify_risk(authenticity_score)
    flags = [e.strip() for _, e in active.values() if e and e.strip()]

    return FusionOutput(
        authenticity_score=authenticity_score,
        risk_classification=risk_classification,
        flags=flags,
    )


def _classify_risk(pct: float) -> str:
    for lower_bound, label in _RISK_BANDS:
        if pct >= lower_bound:
            return label
    return "HIGH"


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Image quality detection + geometry-boost fusion
# ══════════════════════════════════════════════════════════════════════════════

# ELA std-dev threshold below which an image is classified as "new" / uncompressed.
# Images with very uniform ELA maps (low std-dev) are freshly generated or
# saved at high quality — making pixel-texture heuristics unreliable and
# geometric landmark analysis more trustworthy.
_ELA_NEW_IMG_THRESHOLD   = 0.04   # std_dev below this  → "new" / HR image
_ELA_LOW_QUAL_THRESHOLD  = 0.20   # std_dev above this  → heavily compressed
_HR_MIN_PIXELS           = 1_000_000   # ≥1 MP (e.g. 1000×1000) considered high-res

# Geometry weight boost applied for high-quality / high-res images.
# For these images the 3D alignment of eyes/nose is the most reliable signal.
_GEOMETRY_BOOST = 0.20   # +20 pp geo weight; taken equally from ELA and model weights


def detect_image_quality(img_path: str) -> str:
    """
    Classify the input image into one of three quality tiers.

    Returns
    -------
    "high_res"  : large image (≥1 MP) with low ELA variance — freshly generated
                  or high-quality camera shot.  Geometric landmarks are most trustworthy.
    "standard"  : typical web/social-media image with moderate compression history.
    "low_quality": heavily compressed or re-encoded image (high ELA variance).
                  Geometry might be corrupted by JPEG blocking artefacts.

    Implementation
    --------------
    1. Read image dimensions for pixel-count heuristic.
    2. Re-save to an in-memory JPEG at quality=90 and compute the std-dev of the
       absolute pixel difference (same operation as screenshot_forensics.ela_score).
    3. Classify based on pixel count and ELA std-dev thresholds.
    """
    try:
        import io
        import numpy as np
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        n_pixels = w * h

        # ELA computation (identical to screenshot_forensics.ela_score internals)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")

        import numpy as np
        ela_map = np.abs(
            np.asarray(img, dtype=np.float32) - np.asarray(recomp, dtype=np.float32)
        )
        ela_norm = ela_map / (ela_map.max() + 1e-6)
        std_dev  = float(np.std(ela_norm.mean(axis=2)))

        if n_pixels >= _HR_MIN_PIXELS and std_dev < _ELA_NEW_IMG_THRESHOLD:
            quality = "high_res"
        elif std_dev >= _ELA_LOW_QUAL_THRESHOLD:
            quality = "low_quality"
        else:
            quality = "standard"

        logger.debug(
            "detect_image_quality: %s  pixels=%d  ela_std=%.4f  → %s",
            img_path, n_pixels, std_dev, quality,
        )
        return quality

    except Exception as exc:
        logger.warning("detect_image_quality failed for %s: %s", img_path, exc)
        return "standard"


def geometry_boost_combine_results(
    image_result: Optional[ModalityResult] = None,
    video_result: Optional[ModalityResult] = None,
    audio_result: Optional[ModalityResult] = None,
    *,
    img_path: Optional[str] = None,
    forensic_breakdown: Optional[dict] = None,
) -> FusionOutput:
    """
    Quality-aware fusion wrapper that boosts the Geometric/Landmark stream
    weight by 20 pp for high-resolution or freshly-generated images.

    Rationale
    ---------
    For "new" or "high-res" images:
      • JPEG blocking artefacts are absent — ELA is unreliable.
      • Screenshot re-compression has not destroyed landmark precision.
      • The 3D geometry of eyes/nose/mouth is the most stable and least
        domain-specific signal available.

    The boost is applied to the *per-image forensic signal weights* inside
    screenshot_forensics.py by re-weighting the geometry component.  The
    modality-level (image vs video vs audio) weights in _WEIGHTS are unchanged;
    this function only adjusts which internal forensic signal is trusted most
    *within* the image modality.

    Parameters
    ----------
    image_result        : (score, explanation) from process_image(), or None
    video_result        : (score, explanation) from process_video(), or None
    audio_result        : (score, explanation) from process_audio(), or None
    img_path            : path to the image file (used for quality detection)
    forensic_breakdown  : optional dict with keys 'ela', 'lbp', 'geometry'
                          (from screenshot_forensics.screenshot_resistant_score).
                          If provided, the geometry-boosted score is recomputed
                          here and used to override image_result's score.

    Returns
    -------
    FusionOutput (same structure as combine_results)
    """
    boosted_image_result = image_result

    # Only attempt boost when we have an image path + forensic breakdown
    if img_path is not None and forensic_breakdown is not None and image_result is not None:
        quality = detect_image_quality(img_path)

        if quality in ("high_res",):
            # Re-weight forensic signals with +20 pp geometry boost
            ela      = forensic_breakdown.get("ela",      0.0)
            lbp      = forensic_breakdown.get("lbp",      0.0)
            geometry = forensic_breakdown.get("geometry", 0.0)

            # Base weights from screenshot_forensics._SIGNAL_WEIGHTS:
            #   ela=0.40, lbp=0.35, geometry=0.25
            # After +20 pp geometry: ela=0.30, lbp=0.25, geometry=0.45
            # (the extra 0.20 is split -0.10 ela, -0.10 lbp)
            boosted_forensic_suspicion = (
                0.30 * ela
                + 0.25 * lbp
                + 0.45 * geometry            # landmark stream boosted
            )
            # Invert suspicion to authenticity: higher suspicion = lower score
            # Blend 50/50 with the original model score to avoid overcorrection
            orig_score, orig_explanation = image_result
            boosted_score = 0.50 * orig_score + 0.50 * (1.0 - boosted_forensic_suspicion)
            boosted_score = float(max(0.0, min(1.0, boosted_score)))

            boosted_image_result = (
                boosted_score,
                f"[GeoBoost+20%% | {quality}] {orig_explanation}",
            )
            logger.info(
                "geometry_boost: quality=%s  orig=%.3f  boosted=%.3f  "
                "geo=%.3f  ela=%.3f  lbp=%.3f",
                quality, orig_score, boosted_score, geometry, ela, lbp,
            )

    return combine_results(
        image_result=boosted_image_result,
        video_result=video_result,
        audio_result=audio_result,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Stream C — Fusion Head
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class DeepFakeFusionHead(nn.Module):
        """
        Stream C — concatenation head for dual-stream deepfake detection.

        Input
        -----
        spatial_feat   : (B, 1280)  from SpatialStream   (EfficientNet-V2-S)
        frequency_feat : (B,  512)  from FrequencyStream (Phase-Spectrum CNN)

        Architecture
        ------------
        cat([spatial, frequency])   →  (B, 1792)
        BatchNorm(1792)
        Dropout(0.50)               ← main regulariser to prevent overfitting
        Linear(1792, 512) + ReLU
        Dropout(0.30)
        Linear(512, 2)              ← logits:  class 0 = fake, class 1 = real

        Parameters
        ----------
        spatial_dim   : int  feature dim from Stream A  (default 1280)
        frequency_dim : int  feature dim from Stream B  (default 512)
        hidden_dim    : int  intermediate hidden size   (default 512)
        num_classes   : int  output classes             (default 2)
        dropout_main  : float  main dropout probability (default 0.50)
        dropout_hidden: float  hidden dropout probability(default 0.30)
        """

        def __init__(
            self,
            spatial_dim:    int   = 1280,
            frequency_dim:  int   = 512,
            hidden_dim:     int   = 512,
            num_classes:    int   = 2,
            dropout_main:   float = 0.50,
            dropout_hidden: float = 0.30,
        ) -> None:
            super().__init__()
            concat_dim = spatial_dim + frequency_dim   # 1792

            self.head = nn.Sequential(
                nn.BatchNorm1d(concat_dim),
                nn.Dropout(p=dropout_main),
                nn.Linear(concat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_hidden),
                nn.Linear(hidden_dim, num_classes),
            )

            self._init_weights()
            logger.info(
                "DeepFakeFusionHead: in=%d  hidden=%d  out=%d  "
                "dropout=(%.2f, %.2f)",
                concat_dim, hidden_dim, num_classes,
                dropout_main, dropout_hidden,
            )

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(
            self,
            spatial_feat:   "torch.Tensor",
            frequency_feat: "torch.Tensor",
        ) -> "torch.Tensor":
            x = torch.cat([spatial_feat, frequency_feat], dim=1)  # (B, 1792)
            return self.head(x)                                    # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PyTorch Lightning Module
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE and _PL_AVAILABLE:

    class DeepFakeV1Module(pl.LightningModule):
        """
        PyTorch Lightning Module for training the dual-stream deepfake detector.

        Streams
        -------
        A (Spatial)   : SpatialStream    — EfficientNet-V2-S, pretrained
        B (Frequency) : FrequencyStream  — Phase-spectrum CNN, trained from scratch
        C (Fusion)    : DeepFakeFusionHead — concat + Dropout(0.5) + classifier

        Hyperparameters (saved to hparams)
        ------------------------------------
        lr                : AdamW learning rate          (default 1e-4)
        weight_decay      : AdamW weight decay           (default 1e-2)
        label_smoothing   : CrossEntropyLoss smoothing   (default 0.05)
        num_epochs        : CosineAnnealingLR T_max      (default 7)

        Training flags
        ---------------
        Mixed precision (fp16) is enabled externally via Trainer(precision=16).
        The checkpoint callback (in train.py) saves  best_deepfake_v1.pth.

        Forward
        -------
        batch → (images, labels, _paths)
        images : (B, 3, 224, 224)
        logits : (B, 2)
        """

        def __init__(
            self,
            lr:             float = 1e-4,
            weight_decay:   float = 1e-2,
            label_smoothing: float = 0.05,
            num_epochs:     int   = 7,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()

            # ── Import streams here to avoid circular imports at module level ──
            from app.ai_engine.spatial import SpatialStream
            from app.ai_engine.noise   import FrequencyStream

            self.stream_a    = SpatialStream(pretrained=True)
            self.stream_b    = FrequencyStream()
            self.fusion_head = DeepFakeFusionHead(
                spatial_dim=SpatialStream.OUT_DIM,
                frequency_dim=FrequencyStream.OUT_DIM,
            )

            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )

        # ── Forward pass ──────────────────────────────────────────────────────
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feat_a = self.stream_a(x)      # (B, 1280)
            feat_b = self.stream_b(x)      # (B,  512)
            logits = self.fusion_head(feat_a, feat_b)  # (B, 2)
            return logits

        # ── Shared step ───────────────────────────────────────────────────────
        def _shared_step(
            self, batch: tuple, stage: str
        ) -> "torch.Tensor":
            images, labels, _ = batch
            logits = self(images)          # (B, 2)
            loss   = self.criterion(logits, labels)

            preds  = logits.argmax(dim=1)
            acc    = (preds == labels).float().mean()

            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage=="train"),
                     on_epoch=True, sync_dist=True)
            self.log(f"{stage}/acc",  acc,  prog_bar=True, on_step=False,
                     on_epoch=True, sync_dist=True)
            return loss

        def training_step(self, batch: tuple, batch_idx: int) -> "torch.Tensor":
            return self._shared_step(batch, "train")

        def validation_step(self, batch: tuple, batch_idx: int) -> "torch.Tensor":
            return self._shared_step(batch, "val")

        def test_step(self, batch: tuple, batch_idx: int) -> "torch.Tensor":
            return self._shared_step(batch, "test")

        # ── Optimizer + scheduler ─────────────────────────────────────────────
        def configure_optimizers(self):
            # Use different LR for pretrained backbone vs scratch-trained heads
            backbone_params = list(self.stream_a.parameters())
            head_params     = (
                list(self.stream_b.parameters())
                + list(self.fusion_head.parameters())
            )
            param_groups = [
                {"params": backbone_params, "lr": self.hparams.lr * 0.1},
                {"params": head_params,     "lr": self.hparams.lr},
            ]

            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.num_epochs,
                eta_min=1e-6,
            )
            return {
                "optimizer":  optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval":  "epoch",
                    "monitor":   "val/loss",
                },
            }

        # ── Convenience: load a saved checkpoint as inference model ───────────
        @classmethod
        def load_for_inference(
            cls,
            checkpoint_path: str,
            device: str = "cpu",
        ) -> "DeepFakeV1Module":
            """
            Load a saved checkpoint and set model to eval mode.

            Usage
            -----
            model = DeepFakeV1Module.load_for_inference(
                "models/ml_model/best_deepfake_v1.pth"
            )
            """
            model = cls.load_from_checkpoint(checkpoint_path, map_location=device)
            model.eval()
            return model

else:                               # pragma: no cover
    class DeepFakeFusionHead:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise ImportError("pip install torch")

    class DeepFakeV1Module:    # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise ImportError("pip install torch pytorch-lightning")