"""
services/file_service.py
========================
Handles raw file I/O and media-type detection.

NOTE (Step 1)
-------------
`save_file` and `detect_file_type` are **stubs**.
Their signatures, return types, and docstrings define the contract that
the real implementations must satisfy in a later step.
Do NOT add business logic here yet.
"""

from fastapi import UploadFile
import os  # ✅ added

# ---------------------------------------------------------------------------
# Known extension → media-type mapping
# (extend this dict as new formats are supported)
# ---------------------------------------------------------------------------
_EXTENSION_MAP: dict[str, str] = {
    # Images
    "jpg": "image",
    "jpeg": "image",
    "png": "image",
    "webp": "image",
    "gif": "image",
    "bmp": "image",
    "tiff": "image",
    # Videos
    "mp4": "video",
    "mov": "video",
    "avi": "video",
    "mkv": "video",
    "webm": "video",
    "flv": "video",
    # Audio
    "mp3": "audio",
    "wav": "audio",
    "flac": "audio",
    "ogg": "audio",
    "aac": "audio",
    "m4a": "audio",
}


async def save_file(file: UploadFile) -> str:
    """
    Persist an uploaded file to storage and return its path.

    Parameters
    ----------
    file : UploadFile
        The multipart file object received by FastAPI.

    Returns
    -------
    str
        Absolute (or relative) path to the saved file.
        Example: "/uploads/abc123_report.mp4"

    Raises
    ------
    IOError
        If the file cannot be written to storage.

    TODO (Step 2)
    -------------
    - Stream `file.read()` to local disk or object storage (S3 / GCS).
    - Generate a collision-safe filename (UUID prefix recommended).
    - Return the final storage path.
    """
    # --- TEMP IMPLEMENTATION FOR MVP (replaces stub) ---
    
    # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Save file
    file_path = os.path.join("uploads", file.filename)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to save file: {e}")

    return file_path


def detect_file_type(filename: str) -> str:
    """
    Infer the media category from a filename's extension.

    Parameters
    ----------
    filename : str
        Original filename as supplied by the client, e.g. "clip.mp4".

    Returns
    -------
    str
        One of ``"image"``, ``"video"``, ``"audio"``, or ``"unknown"``.

    Notes
    -----
    Extension-based detection is intentionally kept simple for Step 1.
    Step 2 should add magic-byte / MIME sniffing for robustness.
    """
    if "." not in filename:
        return "unknown"

    ext = filename.rsplit(".", 1)[-1].lower()
    return _EXTENSION_MAP.get(ext, "unknown")