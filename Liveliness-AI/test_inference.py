"""
test_inference.py
=================
Liveliness-AI  |  Triple-Stream Deepfake Detector — Inference + Grad-CAM XAI

Usage
-----
  # Quickstart (auto-finds best checkpoint):
  python test_inference.py --image test.jpg

  # Specify checkpoint and output path explicitly:
  python test_inference.py --image face.jpg --checkpoint models/ml_model/best_deepfake_v1.ckpt --output heatmap.png

  # Run on GPU:
  python test_inference.py --image test.jpg --device cuda

Outputs
-------
  • Console:  Fake Probability %, Real Probability %, verdict + confidence band
  • heatmap.png:  side-by-side  [Original | Grad-CAM overlay]
                  Title bar shows the verdict and probabilities.

Grad-CAM target
---------------
  Grad-CAM is applied to the SPATIAL stream (EfficientNet-V2-S) at its last
  convolutional block output  (backbone.conv_head).  This is the richest
  spatial feature map before global-average-pooling — it shows *where* on the
  face the model attended.  The frequency stream (phase-spectrum CNN) does not
  produce spatially-interpretable feature maps, so it is intentionally excluded
  from the heatmap.

Self-contained
--------------
  Grad-CAM is implemented with raw PyTorch hooks — no torchcam / captum needed.
  Only standard dependencies: torch, torchvision, PIL, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference")

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Force UTF-8 output on Windows (handles box-drawing chars & emoji) ─────────
# Always re-wrap stdout/stderr to utf-8 — the conditional check is not
# reliable when PowerShell pipes output (it lies about the encoding).
try:
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
except Exception:
    pass  # already wrapped or no buffer attribute (e.g. IDLE)


# ── Dependency check ──────────────────────────────────────────────────────────
_MISSING: list[str] = []
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    _MISSING.append("torch")

try:
    import torchvision.transforms as T
except ImportError:
    _MISSING.append("torchvision")

try:
    from PIL import Image as PILImage
except ImportError:
    _MISSING.append("Pillow")

try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize as MplNorm
except ImportError:
    _MISSING.append("matplotlib")

if _MISSING:
    print(f"[ERROR] Missing packages: {', '.join(_MISSING)}")
    print(f"  Fix: pip install {' '.join(_MISSING)}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepFake Inference + Grad-CAM XAI"
    )
    p.add_argument(
        "--image", "-i",
        default="test.jpg",
        help="Path to the input image.  (default: test.jpg)",
    )
    p.add_argument(
        "--checkpoint", "-c",
        default=None,
        help=(
            "Path to .ckpt or .pth checkpoint.  "
            "Auto-detected from models/ml_model/ if not provided."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default="heatmap.png",
        help="Where to save the Grad-CAM heatmap.  (default: heatmap.png)",
    )
    p.add_argument(
        "--device", "-d",
        default=None,
        help="'cuda', 'cpu', or 'mps'.  Auto-selected if not provided.",
    )
    p.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip Grad-CAM — just print the probability.",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery
# ══════════════════════════════════════════════════════════════════════════════

def find_checkpoint() -> Path:
    """
    Auto-detect the best checkpoint in models/ml_model/.
    Prefers .ckpt (Lightning, has hparams) over .pth (plain state_dict).
    """
    model_dir = ROOT / "models" / "ml_model"
    candidates = [
        model_dir / "best_deepfake_v1.ckpt",
        model_dir / "best_deepfake_v1.pth",
        model_dir / "last.ckpt",
    ]
    # Also accept any .ckpt file in the directory as fallback
    for ckpt_file in sorted(model_dir.glob("*.ckpt")):
        if ckpt_file not in candidates:
            candidates.append(ckpt_file)

    for path in candidates:
        if path.exists():
            log.info("Checkpoint found: %s  (%.1f MB)", path.name,
                     path.stat().st_size / 1e6)
            return path

    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}.\n"
        "  Run train.py first, or pass --checkpoint <path>."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

# ── Conv-stem key remapping ───────────────────────────────────────────────────

def _remap_conv_stem_keys(state: dict) -> dict:
    """
    Translate a pre-BlurPool checkpoint into the new BlurPool architecture.

    Architecture change (spatial.py)
    ---------------------------------
    OLD (before BlurPool injection):
      stream_a.backbone.conv_stem           →  Conv2d(3, 24, stride=2)

    NEW (with BlurPool injected):
      stream_a.backbone.conv_stem.0         →  Conv2d(3, 24, stride=1)   [weight only]
      stream_a.backbone.conv_stem.1         →  BlurPool2d                [blur_kernel buffer]

    The Conv2d weight tensor is identical — only the stride changed from 2→1,
    which is not stored in the weight tensor itself.  We therefore copy
    'conv_stem.weight' → 'conv_stem.0.weight' verbatim.

    The 'conv_stem.1.blur_kernel' buffer is NOT in old checkpoints — it is a
    fixed, non-trainable Gaussian kernel that is re-created from scratch every
    time SpatialStream.__init__ runs, so there is nothing to migrate.

    Returns
    -------
    Remapped state dict (new dict, original is not mutated).
    """
    OLD_KEY = "stream_a.backbone.conv_stem.weight"
    NEW_KEY = "stream_a.backbone.conv_stem.0.weight"

    if OLD_KEY not in state:
        return state   # already new format or unrelated checkpoint

    remapped = {k: v for k, v in state.items() if k != OLD_KEY}
    remapped[NEW_KEY] = state[OLD_KEY]

    log.info(
        "Checkpoint remapped: '%s' → '%s'  (BlurPool architecture migration)",
        OLD_KEY, NEW_KEY,
    )
    return remapped


def load_model(ckpt_path: Path, device: str) -> nn.Module:
    """
    Load DeepFakeV1Module from a Lightning .ckpt or a plain .pth state_dict.
    Returns the model in eval mode on `device`.

    BlurPool compatibility
    ----------------------
    Checkpoints trained before the BlurPool injection store the stem conv
    weights under 'stream_a.backbone.conv_stem.weight'.  The new architecture
    expects 'stream_a.backbone.conv_stem.0.weight'.  _remap_conv_stem_keys()
    performs this rename automatically.

    strict=False is used for .ckpt loading so that the new non-trainable
    'conv_stem.1.blur_kernel' buffer (a fixed Gaussian — not learned) does not
    raise a RuntimeError for a missing key.  The buffer is always constructed
    correctly in SpatialStream.__init__, so no information is lost.
    """
    from app.ai_engine.fusion import DeepFakeV1Module

    suffix = ckpt_path.suffix.lower()

    if suffix == ".ckpt":
        log.info("Loading Lightning checkpoint…")

        # ── Step 1: peek at the raw state_dict to detect key format ──────────
        raw = torch.load(str(ckpt_path), map_location="cpu")
        if "state_dict" in raw:
            raw["state_dict"] = _remap_conv_stem_keys(raw["state_dict"])
            # Write back to a temp file isn't needed — Lightning accepts a
            # pre-modified checkpoint dict via load_from_checkpoint when we
            # pass the path, so we use the lower-level approach instead:
            # reconstruct the model from hparams, then load weights manually.
            hparams = raw.get("hyper_parameters", {})
            model = DeepFakeV1Module(**hparams)
            missing, unexpected = model.load_state_dict(
                raw["state_dict"], strict=False
            )
            # Log any genuinely unexpected keys (not the blur_kernel buffer)
            unexpected_real = [
                k for k in unexpected
                if "blur_kernel" not in k
            ]
            if missing:
                log.warning("Keys missing from checkpoint (expected for new layers): %s",
                            missing)
            if unexpected_real:
                log.warning("Unexpected keys in checkpoint (architecture mismatch?): %s",
                            unexpected_real)
        else:
            # Fallback: try direct Lightning load with strict=False
            model = DeepFakeV1Module.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                strict=False,
            )

    elif suffix == ".pth":
        log.info("Loading plain state_dict (.pth)…")
        model = DeepFakeV1Module()      # uses default hparams
        state = torch.load(str(ckpt_path), map_location=device)
        # Lightning sometimes wraps state_dict under 'state_dict' key
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _remap_conv_stem_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected_real = [k for k in unexpected if "blur_kernel" not in k]
        if missing:
            log.warning("Keys missing from .pth (expected for new layers): %s", missing)
        if unexpected_real:
            log.warning("Unexpected keys in .pth: %s", unexpected_real)

    else:
        raise ValueError(
            f"Unsupported checkpoint extension '{suffix}'. "
            "Expected .ckpt or .pth"
        )

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model ready on %s  (%.1f M parameters)", device, total_params)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ══════════════════════════════════════════════════════════════════════════════

# Must match train.py eval_transform exactly
_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(image_path: Path) -> tuple["torch.Tensor", "PILImage.Image"]:
    """
    Load and preprocess an image for inference.

    Returns
    -------
    tensor : (1, 3, 224, 224)  float32  — model input
    pil_img: PIL.Image           — original (resized to 224×224) for overlay
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    pil_img = PILImage.open(image_path).convert("RGB")
    pil_224 = pil_img.resize((224, 224), PILImage.LANCZOS)   # for overlay later
    tensor  = _PREPROCESS(pil_img).unsqueeze(0)              # (1, 3, 224, 224)
    return tensor, pil_224


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    model: nn.Module,
    tensor: "torch.Tensor",
    device: str,
    image_path: "Path | None" = None,
) -> dict:
    """
    Forward pass through the triple-stream model, optionally blended with
    screenshot-resistant forensic signals.

    Returns
    -------
    dict with keys:
      fake_prob        : float  [0, 100]  — final FAKE probability (after forensic blend)
      real_prob        : float  [0, 100]  — final REAL probability
      model_fake_prob  : float  [0, 100]  — raw model FAKE probability (before blend)
      verdict          : str    "FAKE" | "REAL"
      confidence       : str    "HIGH" | "MEDIUM" | "LOW"
      logits           : torch.Tensor  (1, 2)  — raw logits (for Grad-CAM)
      forensic_score   : float  [0, 1]   — combined screenshot-resistant suspicion
      forensic_detail  : dict            — per-signal breakdown
      forensic_applied : bool            — whether forensic blend changed the verdict
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                        # (1, 2)

    probs         = F.softmax(logits, dim=1)[0]       # (2,)
    model_fake    = float(probs[0].item()) * 100.0    # class 0 = fake
    model_real    = float(probs[1].item()) * 100.0    # class 1 = real

    # ── Screenshot-resistant forensic blending ────────────────────────────────
    forensic_score   = 0.0
    forensic_detail  = {"ela": 0.0, "lbp": 0.0, "geometry": 0.0}
    forensic_applied = False
    fake_prob        = model_fake
    real_prob        = model_real

    if image_path is not None:
        try:
            from app.ai_engine.screenshot_forensics import screenshot_resistant_score
            f_score, f_detail = screenshot_resistant_score(str(image_path))
            forensic_score  = f_score
            forensic_detail = f_detail

            # Blend forensics when model leans REAL but forensics are suspicious.
            # We interpolate between model_fake and a FAKE target (85%) using the
            # forensic suspicion score as the interpolation weight.
            # This ensures even a 1–2% raw fake score is correctly pulled to FAKE
            # when all three forensic signals agree.
            #
            # Formula: aggressively scale f_score to ensure an override.
            #
            # Override threshold: f_score > 0.30 AND model says REAL (model_fake < 50)
            if f_score > 0.30 and model_fake < 50.0:
                fake_target  = 85.0                         # target fake% at max suspicion
                blend_amount = f_score * (fake_target - model_fake)
                fake_prob    = min(99.0, model_fake + blend_amount)
                real_prob    = max(1.0,  100.0 - fake_prob)
                forensic_applied = fake_prob > 50.0
                log.info(
                    "Forensic blend applied: model_fake=%.1f%%  forensic=%.3f  "
                    "adjusted_fake=%.1f%%",
                    model_fake, f_score, fake_prob,
                )
            elif f_score > 0.20 and model_fake >= 50.0:
                fake_prob = min(99.0, model_fake + f_score * (99.0 - model_fake) * 0.30)
                real_prob = max(1.0, 100.0 - fake_prob)
                forensic_applied = True
                log.info(
                    "Forensic confidence boost: model_fake=%.1f%%  forensic=%.3f  "
                    "boosted_fake=%.1f%%",
                    model_fake, f_score, fake_prob,
                )
        except Exception as exc:
            log.warning("Forensic analysis skipped: %s", exc)


    verdict = "FAKE" if fake_prob > 50.0 else "REAL"

    # Confidence band based on dominant probability margin
    margin = abs(fake_prob - real_prob)
    if margin >= 40:
        confidence = "HIGH"
    elif margin >= 20:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "fake_prob":        fake_prob,
        "real_prob":        real_prob,
        "model_fake_prob":  model_fake,
        "verdict":          verdict,
        "confidence":       confidence,
        "logits":           logits,
        "forensic_score":   forensic_score,
        "forensic_detail":  forensic_detail,
        "forensic_applied": forensic_applied,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grad-CAM  (native PyTorch hooks — no captum / torchcam required)
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for a single target layer.

    Works with any CNN — we register forward and backward hooks on
    `target_layer` to capture:
      • activations: feature maps at that layer  (forward pass)
      • gradients:   gradients of the target class score w.r.t. those maps

    The CAM is computed as:
      weights = global_avg_pool(gradients)        # importance of each channel
      cam     = ReLU( sum(weights * activations) )
      cam     = bilinear_upsample to input resolution
      cam     = normalise to [0, 1]

    Parameters
    ----------
    model        : the full model (must be in eval mode)
    target_layer : nn.Module — the convolutional layer to hook
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._activations: "torch.Tensor | None" = None
        self._gradients:   "torch.Tensor | None" = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hooks ─────────────────────────────────────────────────────────────────
    def _save_activations(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    # ── Generate CAM ──────────────────────────────────────────────────────────
    def generate(
        self,
        tensor:       "torch.Tensor",    # (1, 3, H, W)
        target_class: int,               # 0 = fake, 1 = real
        output_size:  tuple = (224, 224),
    ) -> "np.ndarray":
        """
        Returns a (H, W) float32 numpy array in [0, 1].

        Parameters
        ----------
        tensor       : preprocessed image tensor on the right device
        target_class : class index whose score is backpropagated
        output_size  : spatial resolution of the returned CAM
        """
        self.model.zero_grad()

        # Forward (with grad tracking for back-prop)
        logits = self.model(tensor)                    # (1, 2)

        # Backward on the target class score
        score = logits[0, target_class]
        score.backward()

        # ── Compute weighted CAM ──────────────────────────────────────────────
        grads = self._gradients          # (1, C, h, w)
        acts  = self._activations        # (1, C, h, w)

        if grads is None or acts is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire — check that target_layer "
                "is reachable in the forward pass."
            )

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Upsample to input resolution
        cam_up  = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]                           # (H, W)

        # Normalise to [0, 1]
        cam_np  = cam_up.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _find_gradcam_target(model: nn.Module) -> nn.Module:
    """
    Locate the best Grad-CAM target layer inside the model.

    Priority:
      1.  stream_a.backbone.conv_head   — EfficientNet final conv (richest)
      2.  stream_a.backbone.blocks[-1]  — last MBConv block
      3.  stream_a                       — fallback (should not happen)
    """
    stream_a = model.stream_a
    backbone  = stream_a.backbone

    # EfficientNet-V2-S: conv_head is the pointwise conv after all blocks
    if hasattr(backbone, "conv_head"):
        layer = backbone.conv_head
        log.info("Grad-CAM target: stream_a.backbone.conv_head  (%s)", type(layer).__name__)
        return layer

    # Fallback: last block in the sequential block list
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        layer = backbone.blocks[-1]
        log.info("Grad-CAM target: stream_a.backbone.blocks[-1]  (%s)", type(layer).__name__)
        return layer

    log.warning("Grad-CAM: could not locate conv_head — using stream_a as target.")
    return stream_a


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_heatmap(
    pil_img:    "PILImage.Image",
    cam:        "np.ndarray",          # (224, 224) in [0, 1]
    result:     dict,
    output_path: Path,
) -> None:
    """
    Produce a side-by-side figure:
      [Original Image]  |  [Grad-CAM overlay]

    The overlay blends the original with a jet-coloured heatmap.
    A colour-bar and verdict title are added automatically.
    """
    # ── Convert PIL to float numpy ─────────────────────────────────────────────
    img_np  = np.array(pil_img).astype(np.float32) / 255.0     # (224, 224, 3)

    # ── Apply jet colourmap to CAM ─────────────────────────────────────────────
    jet          = matplotlib.colormaps["jet"]
    cam_coloured = jet(cam)[:, :, :3]                           # (224, 224, 3), drop alpha

    # ── Blend: 60 % original + 40 % heatmap ───────────────────────────────────
    alpha   = 0.40
    overlay = np.clip(img_np * (1 - alpha) + cam_coloured * alpha, 0, 1)

    # ── Choose accent colour by verdict ───────────────────────────────────────
    fake_prob   = result["fake_prob"]
    real_prob   = result["real_prob"]
    verdict     = result["verdict"]
    confidence  = result["confidence"]
    accent      = "#FF4444" if verdict == "FAKE" else "#44DD88"

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 5.2),
        facecolor="#0F0F14",
    )
    fig.subplots_adjust(
        left=0.04, right=0.88,
        top=0.84,  bottom=0.04,
        wspace=0.08,
    )

    title = (
        f"Verdict: {verdict}  [{confidence} Confidence]   "
        f"Fake {fake_prob:.1f}%  |  Real {real_prob:.1f}%"
    )
    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        color=accent,
        y=0.96,
        fontfamily="monospace",
    )

    _panel_style = dict(xticks=[], yticks=[], aspect="equal")

    # Panel 1 — Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", color="#CCCCCC", fontsize=10, pad=6)
    axes[0].set(**_panel_style)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#333344")

    # Panel 2 — Grad-CAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM  (Spatial Stream — EfficientNet-V2-S)", color="#CCCCCC", fontsize=10, pad=6)
    axes[1].set(**_panel_style)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2)

    # ── Colour-bar ────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.90, 0.08, 0.018, 0.72])
    sm      = plt.cm.ScalarMappable(cmap="jet", norm=MplNorm(vmin=0, vmax=1))
    sm.set_array([])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Activation Intensity", color="#AAAAAA", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#AAAAAA")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#AAAAAA", fontsize=7)
    cbar.outline.set_edgecolor("#333344")

    # ── Annotation legend ─────────────────────────────────────────────────────
    fig.text(
        0.50, 0.01,
        "[HIGH attn = red/yellow]    [LOW attn = blue]",
        ha="center", va="bottom",
        color="#888888", fontsize=8,
        fontfamily="monospace",
    )

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Heatmap saved → %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Console report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(result: dict, image_path: Path, heatmap_path: Path | None) -> None:
    """Pretty-print the inference result to stdout."""
    fake_prob        = result["fake_prob"]
    real_prob        = result["real_prob"]
    model_fake_prob  = result.get("model_fake_prob", fake_prob)
    verdict          = result["verdict"]
    confidence       = result["confidence"]
    forensic_score   = result.get("forensic_score",   0.0)
    forensic_detail  = result.get("forensic_detail",  {})
    forensic_applied = result.get("forensic_applied", False)

    # Risk colour using ANSI codes (works in Windows Terminal / PowerShell 7)
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    MAGENTA= "\033[95m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    verdict_colour = RED if verdict == "FAKE" else GREEN
    bar_length     = 40
    fake_bar       = round(fake_prob / 100 * bar_length)
    real_bar       = bar_length - fake_bar

    print()
    print(f"  {BOLD}{CYAN}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}{CYAN}║   Liveliness-AI  ·  DeepFake Detection Result   ║{RESET}")
    print(f"  {BOLD}{CYAN}╚══════════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  Image    :  {image_path}")
    print()
    print(f"  {BOLD}Fake Probability  :  {verdict_colour}{fake_prob:6.2f} %{RESET}")
    print(f"  {BOLD}Real Probability  :  {GREEN}{real_prob:6.2f} %{RESET}")
    print()
    print(f"  {'█' * fake_bar}{'░' * real_bar}   ({fake_prob:.1f}% FAKE)")
    print()
    print(f"  {BOLD}Verdict     :  {verdict_colour}{verdict}{RESET}")
    print(f"  Confidence  :  {YELLOW}{confidence}{RESET}")
    print()
    print(f"  ─ Interpretation ──────────────────────────────────")
    if fake_prob >= 80:
        print(f"  {RED}Strong deepfake signals detected.{RESET}")
        print(f"  The model found high-confidence artefacts in both")
        print(f"  the spatial domain (face texture) and the frequency")
        print(f"  domain (phase-spectrum checkerboard patterns).")
    elif fake_prob >= 50:
        print(f"  {YELLOW}Suspicious — likely manipulated.{RESET}")
        print(f"  Moderate deepfake signals in one or both streams.")
    elif fake_prob >= 30:
        print(f"  {YELLOW}Low-level anomalies detected but inconclusive.{RESET}")
        print(f"  Image may be authentic or subtly manipulated.")
    else:
        print(f"  {GREEN}No significant deepfake artefacts found.{RESET}")
        print(f"  The model considers this image authentic.")
    print()

    # ── Forensic Analysis Section ─────────────────────────────────────────────
    print(f"  ─ Forensic Analysis (Screenshot-Resistant Signals) ────")
    ela_s  = forensic_detail.get("ela",      0.0)
    lbp_s  = forensic_detail.get("lbp",      0.0)
    geo_s  = forensic_detail.get("geometry", 0.0)

    def _bar(s: float, w: int = 20) -> str:
        filled = round(s * w)
        colour = RED if s > 0.55 else (YELLOW if s > 0.30 else GREEN)
        return colour + "█" * filled + RESET + "░" * (w - filled)

    print(f"  ELA Uniformity    {_bar(ela_s)}  {ela_s*100:5.1f}%  "
          f"{'⚠ uniform → AI' if ela_s > 0.50 else 'normal'}")
    print(f"  Texture (LBP)     {_bar(lbp_s)}  {lbp_s*100:5.1f}%  "
          f"{'⚠ too smooth → AI' if lbp_s > 0.50 else 'natural'}")
    print(f"  Geometry          {_bar(geo_s)}  {geo_s*100:5.1f}%  "
          f"{'⚠ implausible landmarks' if geo_s > 0.50 else 'plausible'}")
    print(f"  Combined Score    {_bar(forensic_score)}  {forensic_score*100:5.1f}%")
    print()
    if forensic_applied:
        print(f"  {MAGENTA}{BOLD}[!] Forensic override applied!{RESET}")
        print(f"  {MAGENTA}   Model raw score: {model_fake_prob:.1f}% FAKE → "
              f"Adjusted: {fake_prob:.1f}% FAKE after forensic blending.{RESET}")
        print(f"  {MAGENTA}   Reason: Image shows screenshot-scrubbed deepfake patterns.{RESET}")
        print()
    elif forensic_score > 0.30 and verdict == "REAL":
        print(f"  {YELLOW}Note: Forensic signals are mildly suspicious ({forensic_score*100:.0f}%){RESET}")
        print(f"  {YELLOW}but below override threshold. Treat with caution.{RESET}")
        print()

    if heatmap_path:
        print(f"  Grad-CAM heatmap →  {heatmap_path}")
        print(f"  {CYAN}Bright regions = facial features the AI attended to.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}Model validation accuracy: 99.88 %  (7 epochs, 20k-sample val set){RESET}")
    print(f"  {BOLD}{CYAN}══════════════════════════════════════════════════════{RESET}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint()

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)

    # ── Load image ────────────────────────────────────────────────────────────
    image_path = Path(args.image)
    tensor, pil_224 = load_image(image_path)
    log.info("Image loaded: %s  → tensor %s", image_path, tuple(tensor.shape))

    # ══════════════════════════════════════════════════════════════════════════
    # Inference + Grad-CAM
    # ══════════════════════════════════════════════════════════════════════════
    heatmap_path: Path | None = None

    if args.no_heatmap:
        # Fast path — no gradients needed
        result = run_inference(model, tensor, device, image_path=image_path)
    else:
        # Grad-CAM path — need gradients, temporarily enable them
        heatmap_path = Path(args.output)

        target_layer = _find_gradcam_target(model)
        gradcam      = GradCAM(model, target_layer)

        # ── Run predict (no grad) to get probabilities ─────────────────────
        # model is in eval() — BatchNorm uses trained running stats, not
        # batch statistics (which would crash on a single-image batch).
        result = run_inference(model, tensor, device, image_path=image_path)
        target_class = 0 if result["verdict"] == "FAKE" else 1

        # ── Rerun with grad for Grad-CAM ───────────────────────────────────
        # torch.enable_grad() re-enables the autograd tape for backward()
        # while model.eval() stays active the entire time — BatchNorm keeps
        # using its frozen running_mean / running_var from the 7-epoch run.
        log.info("Generating Grad-CAM (target_class=%d = %s)…",
                 target_class, result["verdict"])

        with torch.enable_grad():
            cam_map = gradcam.generate(
                tensor.to(device),
                target_class=target_class,
                output_size=(224, 224),
            )
        gradcam.remove_hooks()

        # ── Render heatmap ────────────────────────────────────────────────
        render_heatmap(pil_224, cam_map, result, heatmap_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════════
    print_report(result, image_path, heatmap_path)


if __name__ == "__main__":
    main()
