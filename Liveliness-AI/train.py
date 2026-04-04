"""
train.py
========
Liveliness-AI  |  Triple-Stream Deepfake Detector — Training Entry Point

Run from the project root:
    python train.py [--epochs N] [--batch-size N] [--workers N] [--no-gpu]

Streams
-------
  A  SpatialStream    (spatial.py)   EfficientNet-V2-S, pretrained ImageNet
  B  FrequencyStream  (noise.py)     Phase-Spectrum CNN, trained from scratch
  C  DeepFakeFusionHead (fusion.py)  Concat + Dropout(0.5) + classifier

Training config
---------------
  Framework  :  PyTorch Lightning
  Optimizer  :  AdamW  (backbone LR = lr×0.1,  heads LR = lr = 1e-4)
  Scheduler  :  CosineAnnealingLR  (T_max = num_epochs)
  Loss       :  CrossEntropyLoss  (label-smoothing = 0.05)
  Augment    :  DegradationTransform on-the-fly for 20 % of training images
  Precision  :  16-bit mixed precision  (2× speed, ~50 % VRAM)
  Epochs     :  5–8  (default 7, fits in 4-hour window on a V100/A10)
  Checkpoint :  best val_acc  →  models/ml_model/best_deepfake_v1.pth

Image preprocessing
-------------------
  Resize  224×224
  ToTensor
  Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

Dataset
-------
  CSVs   : uploads/train.csv, uploads/valid.csv, uploads/test.csv
  Images : uploads/train/{real,fake}/, uploads/valid/..., uploads/test/...
  Run dataset_setup.py first if images are not yet extracted.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# ── Setup logging early ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ── Resolve project paths ─────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
UPLOADS   = ROOT / "uploads"
MODEL_DIR = ROOT / "models" / "ml_model"
CKPT_PATH = MODEL_DIR / "best_deepfake_v1.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Triple-Stream DeepFake Detector — Training"
    )
    p.add_argument("--epochs",     type=int,   default=7,
                   help="Number of training epochs (default 7).")
    p.add_argument("--batch-size", type=int,   default=32,
                   help="Batch size per GPU (default 32).")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader worker processes (default 4).")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Base AdamW learning rate (default 1e-4).")
    p.add_argument("--no-gpu",     action="store_true",
                   help="Disable GPU even if available.")
    p.add_argument("--precision",  type=int,   default=16, choices=[16, 32],
                   help="Float precision: 16 (mixed, 2x faster) or 32 (default 16).")
    p.add_argument("--fast-dev",   action="store_true",
                   help="Run 1 batch of train+val to verify setup, then exit.")
    p.add_argument("--finetune",   action="store_true",
                   help=(
                       "After main training, run a 3-epoch partial-unfreeze "
                       "fine-tune phase (last 2 EfficientNet blocks, LR=1e-5). "
                       "Improves domain generalisation. Requires the main run "
                       "to complete first."
                   ))
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Data module
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(args: argparse.Namespace):
    """Create train / val / test DataLoaders from the CSV-backed dataset."""
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from app.ai_engine.spatial import DeepFakeDataset, DegradationTransform

    # ImageNet statistics
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # ── Multi-scale JPEG compression helper ───────────────────────────────────
    class MultiScaleJPEGCompression:
        """
        Applies JPEG compression at a quality sampled uniformly from
        [lo, hi] to simulate the full spectrum of compression quality
        found in wild images — from pristine camera RAW (q=95) down to
        heavily compressed social-media reposts (q=25).

        This forces the model to ignore JPEG blocking artefacts as a
        reliable signal (they are not — they depend entirely on the
        social-media platform the image passed through, not on whether
        it is real or fake).
        """
        def __init__(self, quality_lo: int = 25, quality_hi: int = 95) -> None:
            import random
            self.lo = quality_lo
            self.hi = quality_hi
            self._rng = random

        def __call__(self, img):
            import io
            quality = self._rng.randint(self.lo, self.hi)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            from PIL import Image
            return Image.open(buf).copy()

    # ── Train transform — domain-generalisation augmentations ────────────────
    # Augmentation strategy: force the model to ignore colour and compression
    # quality, which are distribution-specific (social-media filters, cameras).
    #
    #   1. ColorJitter: randomly perturbs {brightness, contrast, saturation, hue}
    #      so the model stops relying on per-dataset colour statistics.
    #
    #   2. RandomGrayscale(p=0.15): randomly throws away all colour information.
    #      This prevents the model from using "too-perfect skin tone" as a proxy
    #      for deepfake detection (GAN skintones are subtly different but
    #      this is lost after grayscale, forcing shape-based reasoning).
    #
    #   3. MultiScaleJPEGCompression: applies random JPEG quality in [25, 95].
    #      The model learns that compression artefacts are NOT a reliable signal.
    #
    #   4. DegradationTransform: existing on-the-fly JPEG / Gaussian blur / screenshot
    #      simulation for 20% of samples (defined in spatial.py).
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(
            brightness=0.3,   # ±30% brightness variation
            contrast=0.3,     # ±30% contrast variation
            saturation=0.2,   # ±20% saturation variation
            hue=0.05,         # ±0.05 hue shift (subtle — avoids unnatural colours)
        ),
        T.RandomGrayscale(p=0.15),        # 15% chance of full desaturation
        MultiScaleJPEGCompression(quality_lo=25, quality_hi=95),
        DegradationTransform(degradation_prob=0.20),   # 20% degraded on-the-fly
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    # ── Val / Test transform — clean, deterministic ───────────────────────────
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = DeepFakeDataset(
        csv_path  = UPLOADS / "train.csv",
        base_dir  = UPLOADS,
        split     = "train",
        transform = train_transform,
        # DegradationTransform already included in train_transform above;
        # augment=False here so it is not double-applied.
        augment   = False,
    )
    val_ds = DeepFakeDataset(
        csv_path  = UPLOADS / "valid.csv",
        base_dir  = UPLOADS,
        split     = "valid",
        transform = eval_transform,
    )
    test_ds = DeepFakeDataset(
        csv_path  = UPLOADS / "test.csv",
        base_dir  = UPLOADS,
        split     = "test",
        transform = eval_transform,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # pin_memory=True gives a non-trivial GPU transfer speed-up (~5-10 %)
    pin = not args.no_gpu

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = pin,
        drop_last   = True,          # avoids BatchNorm issues with final tiny batch
        persistent_workers = (args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = pin,
        persistent_workers = (args.workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = pin,
        persistent_workers = (args.workers > 0),
    )

    log.info(
        "DataLoaders ready —  train=%d  val=%d  test=%d  batch=%d",
        len(train_ds), len(val_ds), len(test_ds), args.batch_size,
    )
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Main training routine
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Dependency check ──────────────────────────────────────────────────────
    try:
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        from pytorch_lightning.loggers import CSVLogger
    except ImportError as exc:
        log.error(
            "Missing dependency: %s\n"
            "  pip install torch torchvision pytorch-lightning timm pandas Pillow",
            exc,
        )
        raise SystemExit(1)

    from app.ai_engine.fusion import DeepFakeV1Module

    # ── GPU config ────────────────────────────────────────────────────────────
    if args.no_gpu or not torch.cuda.is_available():
        accelerator = "cpu"
        devices     = 1
        precision   = 32          # fp16 is GPU-only in Lightning
        log.info("Using CPU (no GPU available or --no-gpu set). Forcing precision=32.")
    else:
        accelerator = "gpu"
        devices     = torch.cuda.device_count()
        precision   = args.precision
        log.info("Using %d GPU(s) — precision=%d-bit mixed.", devices, precision)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepFakeV1Module(
        lr             = args.lr,
        weight_decay   = 1e-2,
        label_smoothing= 0.05,
        num_epochs     = args.epochs,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath   = str(MODEL_DIR),
        filename  = "best_deepfake_v1",                 # → best_deepfake_v1.ckpt
        monitor   = "val/acc",
        mode      = "max",
        save_top_k= 1,
        save_last = True,
        verbose   = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Logger (CSV — no WandB account required) ──────────────────────────────
    csv_logger = CSVLogger(
        save_dir = str(ROOT / "logs"),
        name     = "deepfake_v1",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs       = args.epochs,
        accelerator      = accelerator,
        devices          = devices,
        precision        = precision,            # 16 = bf16/fp16 auto-mixed
        callbacks        = [checkpoint_cb, lr_monitor],
        logger           = csv_logger,
        log_every_n_steps= 50,
        gradient_clip_val= 1.0,                 # stabilise training
        fast_dev_run     = args.fast_dev,       # 1-batch smoke-test flag
        deterministic    = False,               # True is slow; keep False for speed
    )

    log.info("=" * 64)
    log.info("  Triple-Stream DeepFake Training  — %d epoch(s)", args.epochs)
    log.info("  Checkpoint will be saved to: %s", CKPT_PATH.parent)
    log.info("=" * 64)

    # ── Fit ───────────────────────────────────────────────────────────────────
    trainer.fit(model, train_loader, val_loader)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    log.info("Running test-set evaluation...")
    best_ckpt = checkpoint_cb.best_model_path
    trainer.test(model, dataloaders=test_loader, ckpt_path=best_ckpt or "best")

    # ── Save as a plain .pth for easy loading without Lightning ───────────────
    _export_plain_pth(best_ckpt, CKPT_PATH)

    log.info("")
    log.info("✔ Training complete.")
    log.info("  Best Lightning checkpoint : %s", best_ckpt)
    log.info("  Plain .pth export         : %s", CKPT_PATH)
    log.info("")
    log.info("Load for inference:")
    log.info("  from app.ai_engine.fusion import DeepFakeV1Module")
    log.info("  model = DeepFakeV1Module.load_for_inference('%s')", CKPT_PATH)

    # ── Optional: Domain generalisation fine-tune phase ─────────────────────────
    if args.finetune:
        _run_finetune_phase(args, best_ckpt, train_loader, val_loader,
                            accelerator, devices, precision)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — domain generalisation fine-tune phase
# ══════════════════════════════════════════════════════════════════════════════

def _run_finetune_phase(
    args,
    best_ckpt_path: str | None,
    train_loader,
    val_loader,
    accelerator: str,
    devices: int,
    precision: int,
) -> None:
    """
    Partial-Unfreeze Fine-Tune Phase
    ---------------------------------
    After the main training run, reload the best checkpoint, freeze the
    entire EfficientNet backbone, then unfreeze only the last 2 MBConv
    blocks (stages 4 and 5).  Re-train for 3 epochs with LR=1e-5.

    Why this helps
    --------------
    The initial training memorises training-set statistics because the
    full backbone encodes dataset-specific texture cues.  Partial-unfreeze
    fine-tuning exposes only the *high-level semantic* layers to new
    gradient updates.  These layers encode shape, geometry, and semantic
    composition — which are consistent across domains — while the low-level
    texture layers (stages 0–3) remain frozen, preventing catastrophic
    forgetting of the generalised features already learned.

    Saved checkpoint: models/ml_model/best_deepfake_v1_finetune.ckpt
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from app.ai_engine.fusion import DeepFakeV1Module

    FINETUNE_EPOCHS = 3
    FINETUNE_LR     = 1e-5
    FINETUNE_CKPT   = MODEL_DIR / "best_deepfake_v1_finetune"

    log.info("=" * 64)
    log.info("  Domain Generalisation Fine-Tune — %d epoch(s) @ LR=%.0e",
             FINETUNE_EPOCHS, FINETUNE_LR)
    log.info("  Unfreezing last 2 EfficientNet blocks (stages 4-5)")
    log.info("=" * 64)

    if not best_ckpt_path:
        log.error("No checkpoint to fine-tune from. Run without --finetune first.")
        return

    # Reload best weights, then apply partial_unfreeze to stream_a
    ft_model = DeepFakeV1Module.load_from_checkpoint(
        best_ckpt_path,
        map_location="cpu",
        # Override hyperparams for fine-tune phase
        lr=FINETUNE_LR,
        num_epochs=FINETUNE_EPOCHS,
    )
    ft_model.stream_a.partial_unfreeze(n_blocks=2)

    # Checkpoint callback — saves best fine-tuned model
    ft_ckpt_cb = ModelCheckpoint(
        dirpath   = str(MODEL_DIR),
        filename  = "best_deepfake_v1_finetune",
        monitor   = "val/acc",
        mode      = "max",
        save_top_k= 1,
        verbose   = True,
    )

    ft_trainer = pl.Trainer(
        max_epochs       = FINETUNE_EPOCHS,
        accelerator      = accelerator,
        devices          = devices,
        precision        = precision,
        callbacks        = [ft_ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        logger           = CSVLogger(str(ROOT / "logs"), name="deepfake_v1_finetune"),
        log_every_n_steps= 50,
        gradient_clip_val= 0.5,    # tighter clip — very low LR, don't overstep
        deterministic    = False,
    )

    ft_trainer.fit(ft_model, train_loader, val_loader)

    best_ft = ft_ckpt_cb.best_model_path
    _export_plain_pth(best_ft, MODEL_DIR / "best_deepfake_v1_finetune.pth")

    log.info("")
    log.info("✔ Fine-tune complete.")
    log.info("  Fine-tuned checkpoint : %s", best_ft)
    log.info("  Use --checkpoint path/to/best_deepfake_v1_finetune.ckpt for inference.")
    log.info("")


# ══════════════════════════════════════════════════════════════════════════════
# Helper — export plain .pth (state_dict only)
# ══════════════════════════════════════════════════════════════════════════════

def _export_plain_pth(best_ckpt_path: str | None, output_path: Path) -> None:
    """
    Extract the model state_dict from the Lightning .ckpt file and save it as
    a plain PyTorch .pth file so it can be loaded without Lightning at inference.
    """
    import torch
    from app.ai_engine.fusion import DeepFakeV1Module

    if not best_ckpt_path:
        log.warning("No best checkpoint found — skipping .pth export.")
        return

    log.info("Exporting plain state_dict → %s", output_path)
    try:
        model = DeepFakeV1Module.load_from_checkpoint(best_ckpt_path, map_location="cpu")
        torch.save(model.state_dict(), str(output_path))
        log.info("✔  Exported: %s  (%.1f MB)",
                 output_path.name, output_path.stat().st_size / 1e6)
    except Exception as exc:
        log.error("Failed to export .pth: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
