"""
app/ai_engine/model_singleton.py
=================================
Liveliness-AI | Global DeepFakeV1Module singleton for the FastAPI server.

Purpose
-------
Guarantees that the EXACT same model instance and preprocessing pipeline
is used for every API request — identical to what test_inference.py runs.

Preprocessing contract (must stay in sync with test_inference.py)
-----------------------------------------------------------------
  1. PIL.Image.open(path).convert("RGB")   — always RGB, never BGR
  2. T.Resize((224, 224))                  — bilinear by default
  3. T.ToTensor()                           — [0, 255] uint8 → [0.0, 1.0] float32
  4. T.Normalize(mean=[0.485, 0.456, 0.406],
                 std =[0.229, 0.224, 0.225])  — ImageNet stats

Key rules
---------
• model.eval() is called once here, globally, at load time.
• inference is always wrapped in torch.inference_mode() by the caller.
• The singleton is loaded during FastAPI lifespan (startup), not on the
  first request, so cold-start latency does not hit the user.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Preprocessing — MUST match test_inference.py _PREPROCESS exactly ─────────
# Any change here must be mirrored in test_inference.py and vice-versa.

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage

    PREPROCESS = T.Compose([
        T.Resize((224, 224)),          # step 1 — resize (bilinear, same as PIL LANCZOS
                                       #           in test_inference — see note below)
        T.ToTensor(),                  # step 2 — [0,255] → [0.0, 1.0], HWC→CHW
        T.Normalize(                   # step 3 — ImageNet normalisation
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
    # NOTE on resize interpolation:
    # test_inference.py does pil_img.resize((224,224), PIL.LANCZOS) only for the
    # *overlay* image; the tensor path also goes through T.Resize which uses
    # bilinear by default.  Both sides use T.Resize((224,224)) on the tensor
    # pipeline, so they are identical.

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch / torchvision not installed — model singleton unavailable.")

# ── Checkpoint discovery ──────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent.parent.resolve()
_MODEL_DIR  = _ROOT / "models" / "ml_model"
_CKPT_CANDIDATES = [
    _MODEL_DIR / "best_deepfake_v1_finetune.ckpt",  # domain-generalised (preferred)
    _MODEL_DIR / "best_deepfake_v1.ckpt",
    _MODEL_DIR / "best_deepfake_v1_finetune.pth",
    _MODEL_DIR / "best_deepfake_v1.pth",
    _MODEL_DIR / "last.ckpt",
]

# ── Singleton state ───────────────────────────────────────────────────────────

_MODEL: Optional["torch.nn.Module"] = None
_DEVICE: str = "cpu"
_LOAD_ERROR: Optional[str] = None


def _find_checkpoint() -> Optional[Path]:
    """Return the best available checkpoint path, or None."""
    for path in _CKPT_CANDIDATES:
        if path.exists():
            return path
    return None


def _remap_conv_stem_keys(state: dict) -> dict:
    """
    Migrate pre-BlurPool checkpoints:
      OLD: stream_a.backbone.conv_stem.weight
      NEW: stream_a.backbone.conv_stem.0.weight
    """
    OLD = "stream_a.backbone.conv_stem.weight"
    NEW = "stream_a.backbone.conv_stem.0.weight"
    if OLD not in state:
        return state
    remapped = {k: v for k, v in state.items() if k != OLD}
    remapped[NEW] = state[OLD]
    logger.info("model_singleton: BlurPool key remapped ('%s' → '%s')", OLD, NEW)
    return remapped


def load_model(device: str = "cpu") -> None:
    """
    Load DeepFakeV1Module into the global singleton.
    Called once during FastAPI lifespan startup.

    Sets model.eval() globally — callers must NOT call model.train().
    """
    global _MODEL, _DEVICE, _LOAD_ERROR

    if not _TORCH_AVAILABLE:
        _LOAD_ERROR = "PyTorch not installed."
        logger.error(_LOAD_ERROR)
        return

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        _LOAD_ERROR = (
            f"No checkpoint found in {_MODEL_DIR}. "
            "Run train.py first."
        )
        logger.error(_LOAD_ERROR)
        return

    logger.info("model_singleton: loading checkpoint %s", ckpt_path.name)

    try:
        from app.ai_engine.fusion import DeepFakeV1Module

        suffix = ckpt_path.suffix.lower()

        if suffix == ".ckpt":
            raw = torch.load(str(ckpt_path), map_location="cpu")
            if "state_dict" in raw:
                raw["state_dict"] = _remap_conv_stem_keys(raw["state_dict"])
                hparams = raw.get("hyper_parameters", {})
                model = DeepFakeV1Module(**hparams)
                missing, unexpected = model.load_state_dict(
                    raw["state_dict"], strict=False
                )
                unexpected_real = [k for k in unexpected if "blur_kernel" not in k]
                if missing:
                    logger.warning("model_singleton: missing keys (new layers): %s", missing)
                if unexpected_real:
                    logger.warning("model_singleton: unexpected keys: %s", unexpected_real)
            else:
                model = DeepFakeV1Module.load_from_checkpoint(
                    str(ckpt_path), map_location="cpu", strict=False
                )

        elif suffix == ".pth":
            model = DeepFakeV1Module()
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _remap_conv_stem_keys(state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            unexpected_real = [k for k in unexpected if "blur_kernel" not in k]
            if missing:
                logger.warning("model_singleton: missing keys: %s", missing)
            if unexpected_real:
                logger.warning("model_singleton: unexpected keys: %s", unexpected_real)

        else:
            raise ValueError(f"Unsupported checkpoint extension: {suffix}")

        # ── Critical: set eval globally, once ────────────────────────────────
        model = model.to(device)
        model.eval()

        _MODEL  = model
        _DEVICE = device

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "model_singleton: ready on %s  (%.1f M params, ckpt=%s)",
            device, n_params, ckpt_path.name,
        )

    except Exception as exc:
        _LOAD_ERROR = f"Model load failed: {exc}"
        logger.exception("model_singleton: %s", _LOAD_ERROR)


def get_model():
    """
    Return the loaded model.  Raises RuntimeError if not yet loaded.
    """
    if _MODEL is None:
        raise RuntimeError(
            _LOAD_ERROR or "Model not loaded. Call load_model() first "
            "(happens automatically during FastAPI lifespan startup)."
        )
    return _MODEL


def get_device() -> str:
    return _DEVICE


def preprocess_image_path(file_path: str) -> "torch.Tensor":
    """
    Load an image from disk and apply the canonical preprocessing pipeline.

    Preprocessing (identical to test_inference.py)
    -----------------------------------------------
    1. PIL.Image.open().convert("RGB")
    2. T.Resize((224, 224))
    3. T.ToTensor()
    4. T.Normalize(ImageNet mean/std)

    Returns
    -------
    torch.Tensor of shape (1, 3, 224, 224) on CPU.
    Debug log prints first 5 values of the flattened tensor so they can be
    compared against the equivalent log line in test_inference.py.
    """
    img = PILImage.open(file_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0)   # (1, 3, 224, 224)

    # ── Pixel-value debug log (parity check) ─────────────────────────────────
    flat5 = tensor[0].flatten()[:5].tolist()
    logger.debug(
        "[PARITY] API preprocessed tensor first 5 values: %s  (path=%s)",
        [f"{v:.6f}" for v in flat5], file_path,
    )
    print(
        f"[PARITY-API] first 5 pixel values: "
        f"{[round(v, 6) for v in flat5]}  | file={file_path}"
    )

    return tensor
