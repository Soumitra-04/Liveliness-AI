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
                   help="Float precision: 16 (mixed, 2× faster) or 32 (default 16).")
    p.add_argument("--fast-dev",   action="store_true",
                   help="Run 1 batch of train+val to verify setup, then exit.")
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

    # ── Train transform — DegradationTransform injected before normalisation ──
    train_transform = T.Compose([
        T.Resize((224, 224)),
        DegradationTransform(degradation_prob=0.20),   # 20 % degraded on-the-fly
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
