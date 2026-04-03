"""
file_handler.py — Save an uploaded file to a temp path and clean up after use.
"""

from __future__ import annotations

import os
import tempfile
import shutil
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import UploadFile

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def validate_extension(filename: str) -> str:
    """
    Validate that `filename` has a supported video extension.

    Returns
    -------
    str
        The lower-cased extension (e.g. ".mp4").

    Raises
    ------
    ValueError
        If the extension is not supported.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )
    return ext


@asynccontextmanager
async def save_upload(upload: UploadFile) -> AsyncGenerator[str, None]:
    """
    Async context manager that saves `upload` to a temp file and yields
    its path.  The temp file is deleted when the context exits.

    Usage
    -----
    async with save_upload(upload) as path:
        process(path)
    """
    _, ext = os.path.splitext(upload.filename or "video.mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext.lower())
    try:
        # Stream the upload to disk in 1 MB chunks
        while chunk := await upload.read(1024 * 1024):
            tmp.write(chunk)
        tmp.close()
        yield tmp.name
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
