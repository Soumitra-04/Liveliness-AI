"""
Targeted Few-Shot Finetuning script
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ── Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("finetune")

# Model and paths
ROOT = Path(r"c:\Users\ishik\OneDrive\Desktop\GDG\Liveliness-AI\Liveliness-AI")
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models" / "ml_model"
CKPT_PATH = MODEL_DIR / "best_deepfake_v1.ckpt"
BAK_PATH  = MODEL_DIR / "best_deepfake_v1.ckpt.bak"
PTH_PATH  = MODEL_DIR / "best_deepfake_v1.pth"
UPLOADS_DIR = ROOT / "database" / "__pycache__" / "uploads"

# Images mapping: {filename: class_index (0=FAKE, 1=REAL)}
SAMPLES = {
    "fake.png": 0,
    "test.jpg": 0,
    "download.jpg": 1,
    "download (2).jpg": 1,
    "download (3).jpg": 1,
}

_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image_tensor(filepath: Path) -> torch.Tensor:
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))
    pil_img = Image.open(filepath).convert("RGB")
    return _PREPROCESS(pil_img).unsqueeze(0)

def main():
    from app.ai_engine.fusion import DeepFakeV1Module
    
    # 1. Back up Checkpoint
    if not BAK_PATH.exists():
        log.info(f"Backing up original checkpoint: {BAK_PATH}")
        shutil.copy2(CKPT_PATH, BAK_PATH)
    else:
        log.info(f"Backup already exists at: {BAK_PATH}")

    device = torch.device("cpu")
    log.info("Loading model...")
    # Load model and disable Strict mode because some keys might be slightly mismatched if loading Lightning into base model
    model = DeepFakeV1Module.load_from_checkpoint(str(CKPT_PATH), map_location=device)
    model.to(device)

    # 2. Freeze everything except fusion_head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fusion_head.parameters():
        param.requires_grad = True

    # 3. Load Data
    tensors = []
    labels = []
    for filename, label in SAMPLES.items():
        filepath = UPLOADS_DIR / filename
        log.info(f"Loading {filename} (Label: {'FAKE' if label==0 else 'REAL'})")
        try:
            tensor = load_image_tensor(filepath)
            tensors.append(tensor)
            labels.append(label)
        except Exception as e:
            log.warning(f"Failed to load {filename}: {e}")

    if not tensors:
        log.error("No images loaded.")
        return

    x = torch.cat(tensors, dim=0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    # 4. Optimization
    optimizer = optim.AdamW(model.fusion_head.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train() # Make sure dropout is active for regularization over few samples
    # For Batch Norm, keeping track of stats is unwanted for so few samples, so ensure backbone elements are eval mode?
    # Actually DeepFakeV1Module uses train() globally which might update BN.
    # We will manually switch backbone to eval
    model.stream_a.eval()
    model.stream_b.eval()

    n_steps = 25
    log.info(f"\nStarting Fast Fine-Tuning for {n_steps} steps...")
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # We manually call forward without no_grad() for entire model, but backprop only to fusion_head
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (step+1) % 5 == 0:
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                acc = (preds == y).float().mean()
                log.info(f"Step {step+1}/{n_steps} - Loss: {loss.item():.4f} - Batch Acc: {acc.item()*100:.1f}%")

    # 5. Save updated checkpoint
    log.info(f"\nSaving updated checkpoints...")
    # Update Lightning ckpt
    ckpt_state = torch.load(str(BAK_PATH), map_location=device)
    ckpt_state['state_dict'] = model.state_dict()
    torch.save(ckpt_state, str(CKPT_PATH))
    # Update plain .pth
    torch.save(model.state_dict(), str(PTH_PATH))
    log.info("Finished fine-tuning successfully!")

if __name__ == "__main__":
    main()
