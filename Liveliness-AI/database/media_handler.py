"""
Liveliness-AI — Step 1: Metadata Handling & Database Storage
Responsibilities:
    - File type detection by extension
    - SQLite database initialization
    - Metadata insertion and retrieval
"""

import sqlite3
import os
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DB_DIR  = os.path.join(os.path.dirname(__file__))
DB_PATH = os.path.join(DB_DIR, "media.db")

# Supported extensions mapped to their media type
EXTENSION_MAP: dict[str, str] = {
    # Images
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    # Videos
    ".mp4":  "video",
    ".avi":  "video",
    ".mov":  "video",
    # Audio
    ".wav":  "audio",
    ".mp3":  "audio",
}


# ─────────────────────────────────────────────
# Part 1 — File Type Detection
# ─────────────────────────────────────────────

def detect_file_type(filename: str) -> str:
    """
    Detect the media type of a file based on its extension.

    Args:
        filename: Name (or path) of the file, e.g. "clip.mp4"

    Returns:
        "image", "video", "audio", or "unknown" if unrecognised.
    """
    _, ext = os.path.splitext(filename)
    media_type = EXTENSION_MAP.get(ext.lower(), "unknown")

    if media_type == "unknown":
        print(f"[WARN] Unrecognised extension '{ext}' for file '{filename}'.")

    return media_type


# ─────────────────────────────────────────────
# Part 2 — Database Initialisation
# ─────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """
    Open (and if necessary create) the SQLite database.
    Returns a connection with Row factory enabled for dict-like access.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    return conn


def init_database() -> None:
    """
    Create the 'media_files' table if it does not already exist.
    Safe to call multiple times — uses IF NOT EXISTS.

    Schema
    ------
    id          INTEGER  — auto-incrementing primary key
    filename    TEXT     — original file name supplied by the caller
    file_type   TEXT     — "image" | "video" | "audio" | "unknown"
    file_path   TEXT     — full or relative path to the stored file
    upload_time TEXT     — ISO-8601 UTC timestamp set at insert time
    status      TEXT     — lifecycle status; initially "saved"
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS media_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT    NOT NULL,
                file_type   TEXT    NOT NULL,
                file_path   TEXT    NOT NULL,
                upload_time TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'saved'
            )
        """)
        conn.commit()
        print(f"[DB] Database ready at: {DB_PATH}")
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to initialise database: {e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Part 3 — Metadata Insertion
# ─────────────────────────────────────────────

def store_metadata(filename: str, file_type: str, file_path: str) -> int | None:
    """
    Insert a media-file record into the database.

    Args:
        filename:  Original file name (e.g. "interview.mp4").
        file_type: Media category returned by detect_file_type().
        file_path: Absolute or relative path where the file is stored.

    Returns:
        The auto-generated integer ID of the inserted row,
        or None if the insert failed.
    """
    upload_time = datetime.now(timezone.utc).isoformat()
    status      = "saved"

    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO media_files (filename, file_type, file_path, upload_time, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, file_type, file_path, upload_time, status),
        )
        conn.commit()
        record_id = cursor.lastrowid
        print(f"[DB] Stored '{filename}' as {file_type} — record ID: {record_id}")
        return record_id

    except sqlite3.Error as e:
        print(f"[ERROR] Failed to store metadata for '{filename}': {e}")
        return None

    finally:
        conn.close()


# ─────────────────────────────────────────────
# Helper — Fetch a record by ID (optional util)
# ─────────────────────────────────────────────

def fetch_by_id(record_id: int) -> dict | None:
    """
    Retrieve a single media record by its primary key.

    Args:
        record_id: The integer ID returned by store_metadata().

    Returns:
        A dict with all columns, or None if not found.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM media_files WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to fetch record {record_id}: {e}")
        return None
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Module bootstrap — init DB on import
# ─────────────────────────────────────────────

# The database (and table) are created the first time this module is imported,
# so callers never need to remember to call init_database() manually.
init_database()