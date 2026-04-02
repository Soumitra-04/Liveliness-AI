"""
services/metadata_service.py
=============================
Handles persistence of file metadata to the database.

NOTE (Step 1)
-------------
`store_metadata` is a **stub**.
Its signature and return contract are fixed; the DB implementation
(SQLAlchemy / async ORM / raw SQL) is deferred to a later step.
"""

from datetime import datetime  # ✅ added


def store_metadata(filename: str, file_type: str, file_path: str) -> dict:
    """
    Persist file metadata and return the created record.

    Parameters
    ----------
    filename : str
        Original filename as uploaded by the client.
    file_type : str
        Media category — one of ``"image"``, ``"video"``, ``"audio"``.
    file_path : str
        Path returned by ``save_file()``.

    Returns
    -------
    dict
        A dictionary representing the newly created DB record.
        **Must** contain at minimum:

        .. code-block:: python

            {
                "id": int,          # auto-incremented primary key
                "filename": str,
                "file_type": str,
                "file_path": str,
                "created_at": str,  # ISO-8601 timestamp (optional but recommended)
            }

    Raises
    ------
    RuntimeError
        If the record cannot be inserted (e.g. DB connection failure).

    TODO (Step 2)
    -------------
    - Connect to the database (PostgreSQL / SQLite / etc.).
    - Insert a row into the `media_files` table.
    - Return the full ORM object or a serialised dict.
    """
    # --- TEMP IMPLEMENTATION FOR MVP (replaces stub) ---
    
    try:
        return {
            "id": 1,  # dummy ID
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "created_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to store metadata: {e}")