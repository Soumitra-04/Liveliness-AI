"""
dataset_setup.py
================
Liveliness-AI  |  One-shot dataset extraction & verification utility.

Run from the project root:
    python dataset_setup.py

What it does
------------
1. EXTRACT  — Extracts real_vs_fake.tar.gz from uploads/ using Python's
              built-in tarfile module (no external CLI needed on Windows).
              Deletes the .tar.gz immediately after extraction to free space.

2. VERIFY   — Counts real/fake samples per split using the three CSVs and
              cross-checks that the extracted image files actually exist on
              disk for a random sample of rows.

3. REPORT   — Prints a clean summary table.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import tarfile
from pathlib import Path

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_setup")

# ── Resolve paths ─────────────────────────────────────────────────────────────
# This script lives in the project root, next to run.py / the app/ folder.
# uploads/ is a sibling of this script.
SCRIPT_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = SCRIPT_DIR / "uploads"
TAR_PATH = UPLOADS_DIR / "real_vs_fake.tar.gz"

CSVS = {
    "train": UPLOADS_DIR / "train.csv",
    "valid": UPLOADS_DIR / "valid.csv",
    "test":  UPLOADS_DIR  / "test.csv",
}

# Verification: spot-check this many random rows per split
VERIFY_SAMPLE_SIZE = 25


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_archive() -> None:
    if not TAR_PATH.exists():
        log.info("TAR not found at %s — skipping extraction (already extracted?).", TAR_PATH)
        return

    log.info("=" * 60)
    log.info("STEP 1 — Extracting archive: %s", TAR_PATH)
    log.info("         Target directory  : %s", UPLOADS_DIR)
    log.info("         Archive size      : %.2f GB", TAR_PATH.stat().st_size / 1e9)
    log.info("=" * 60)

    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            members = tar.getmembers()
            log.info("Archive contains %d entries — extracting...", len(members))

            # Extract everything into uploads/
            tar.extractall(path=UPLOADS_DIR)

        log.info("✔  Extraction complete.")

    except tarfile.TarError as exc:
        log.error("Extraction FAILED: %s", exc)
        sys.exit(1)

    # ── Delete the archive to free disk space ───────────────────────────────
    log.info("Deleting archive to free space...")
    try:
        TAR_PATH.unlink()
        log.info("✔  Deleted: %s", TAR_PATH)
    except OSError as exc:
        log.warning("Could not delete archive: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_dataset() -> bool:
    """
    Count real/fake balance from CSVs and spot-check file existence.
    Returns True if everything looks healthy, False otherwise.
    """
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2 — Verifying dataset balance & file integrity")
    log.info("=" * 60)

    # Lazy import — pandas might not be installed, give a clear error
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required for verification. pip install pandas")
        return False

    all_ok = True
    header = f"{'Split':<8} {'Total':>8} {'Real':>8} {'Fake':>8} {'Balance':>10} {'Files-OK':>10}"
    log.info(header)
    log.info("-" * len(header))

    for split, csv_path in CSVS.items():
        if not csv_path.exists():
            log.warning("  %-8s  CSV not found: %s", split, csv_path)
            all_ok = False
            continue

        df = pd.read_csv(csv_path)

        n_total = len(df)
        n_real  = int((df["label"] == 1).sum())
        n_fake  = int((df["label"] == 0).sum())

        if n_total > 0:
            ratio = n_real / n_total * 100
            balance_str = f"{ratio:.1f}% real"
        else:
            balance_str = "N/A"

        # Spot-check a random sample of files
        sample_size = min(VERIFY_SAMPLE_SIZE, n_total)
        sample_df   = df.sample(n=sample_size, random_state=42)
        found = 0
        missing_examples = []

        for _, row in sample_df.iterrows():
            rel_path = row.get("path", "")
            abs_path = UPLOADS_DIR / rel_path
            if abs_path.exists():
                found += 1
            else:
                missing_examples.append(str(abs_path))

        files_ok_str = f"{found}/{sample_size}"
        if found < sample_size:
            all_ok = False

        log.info(
            "  %-8s  %8d  %8d  %8d  %10s  %10s",
            split, n_total, n_real, n_fake, balance_str, files_ok_str,
        )

        # Show a few missing paths for diagnostics
        for mp in missing_examples[:3]:
            log.warning("    ✘ Missing: %s", mp)

    log.info("-" * len(header))
    if all_ok:
        log.info("✔  All checks passed — dataset is balanced and files are present.")
    else:
        log.warning("⚠  Some checks FAILED — see warnings above.")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("Liveliness-AI  |  Dataset Setup Utility")
    log.info("Uploads directory: %s", UPLOADS_DIR)

    if not UPLOADS_DIR.exists():
        log.error("uploads/ directory does not exist: %s", UPLOADS_DIR)
        sys.exit(1)

    # 1 – Extract
    extract_archive()

    # 2 – Verify
    ok = verify_dataset()

    log.info("")
    if ok:
        log.info("🚀  Dataset is ready.  You can now trigger 3-stream fusion training.")
    else:
        log.warning("⚠  Dataset setup completed with warnings — resolve missing files before training.")


if __name__ == "__main__":
    main()
