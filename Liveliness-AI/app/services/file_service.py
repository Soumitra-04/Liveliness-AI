"""
Liveliness-AI — File Storage Module
====================================
Handles uploaded media file saving for the deepfake detection pipeline.
Step 1: File upload and storage pipeline.

Usage:
    from file_storage import save_file
    result = await save_file(upload_file)
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEMP_DIR = Path("temp")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("liveliness_ai.file_storage")

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SavedFile:
    """Holds metadata about a successfully saved file."""
    filename: str       # Unique filename on disk  (e.g. "abc123_video.mp4")
    file_path: str      # Full path string         (e.g. "temp/abc123_video.mp4")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_temp_dir(directory: Path) -> None:
    """
    Create the temp directory (and any parents) if it does not already exist.
    Uses exist_ok=True so concurrent calls never raise a race-condition error.
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("Temp directory ready: %s", directory.resolve())


def _build_unique_filename(original_filename: str) -> str:
    """
    Prepend a UUID4 hex prefix to the original filename so that repeated
    uploads of the same file never collide.

    Example:
        "video.mp4"  →  "3f8a1c2d_video.mp4"
    """
    # Use only the first 8 hex chars — short but statistically collision-free
    unique_prefix = uuid.uuid4().hex[:8]
    # Sanitise the original name to strip any leading path separators
    safe_name = Path(original_filename).name
    return f"{unique_prefix}_{safe_name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def save_file(file) -> SavedFile:
    """
    Persist an uploaded media file to the temp directory.

    Parameters
    ----------
    file : UploadFile-compatible object
        Must expose:
          • file.filename  (str)  — original filename provided by the client
          • file.file      (IO)   — file-like object (SpooledTemporaryFile, etc.)

    Returns
    -------
    SavedFile
        .filename  — unique filename written to disk
        .file_path — path string relative to CWD (e.g. "temp/abc_video.mp4")

    Raises
    ------
    ValueError
        If the uploaded file object or its filename is missing / empty.
    IOError
        If writing to disk fails for any reason.
    """

    # ------------------------------------------------------------------
    # 1. Validate the incoming file object
    # ------------------------------------------------------------------
    if file is None:
        raise ValueError("No file was provided to save_file().")

    if not getattr(file, "filename", None):
        raise ValueError("Uploaded file has no filename; cannot save.")

    original_name: str = file.filename
    logger.info("Received file for saving: '%s'", original_name)

    # ------------------------------------------------------------------
    # 2. Guarantee the temp/ directory exists before writing
    # ------------------------------------------------------------------
    _ensure_temp_dir(TEMP_DIR)

    # ------------------------------------------------------------------
    # 3. Build a collision-free destination path
    # ------------------------------------------------------------------
    unique_name = _build_unique_filename(original_name)
    destination = TEMP_DIR / unique_name

    logger.info("Saving to: %s", destination)

    # ------------------------------------------------------------------
    # 4. Stream the file to disk using shutil.copyfileobj
    #    This avoids loading the entire file into memory at once,
    #    which matters for large video files.
    # ------------------------------------------------------------------
    try:
        with destination.open("wb") as out_file:
            # Reset the read cursor in case the file was partially read upstream
            await _seek_if_possible(file)
            shutil.copyfileobj(file.file, out_file)

    except OSError as exc:
        # Clean up any partially written file so we don't leave junk on disk
        if destination.exists():
            destination.unlink(missing_ok=True)
            logger.warning("Partial file removed after write failure: %s", destination)
        raise IOError(
            f"Failed to write '{unique_name}' to '{TEMP_DIR}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 5. Confirm the file was actually written (non-zero size guard)
    # ------------------------------------------------------------------
    written_bytes = destination.stat().st_size
    if written_bytes == 0:
        destination.unlink(missing_ok=True)
        raise IOError(
            f"File '{original_name}' was saved but appears empty (0 bytes). "
            "Upload may be corrupted."
        )

    logger.info(
        "File saved successfully — name: '%s', size: %d bytes, path: '%s'",
        unique_name,
        written_bytes,
        destination,
    )

    # ------------------------------------------------------------------
    # 6. Return structured metadata for the calling API layer
    # ------------------------------------------------------------------
    return SavedFile(
        filename=unique_name,
        file_path=str(destination),
    )


# ---------------------------------------------------------------------------
# Internal async helper
# ---------------------------------------------------------------------------

async def _seek_if_possible(file) -> None:
    """
    Attempt to seek the underlying file back to position 0.
    Some ASGI frameworks partially read the file during validation;
    this ensures we always copy from the beginning.
    Silently skips if the stream is not seekable.
    """
    try:
        underlying = file.file
        if hasattr(underlying, "seek"):
            underlying.seek(0)
    except Exception:
        pass  # Non-seekable streams (e.g. raw sockets) are fine — just proceed

 