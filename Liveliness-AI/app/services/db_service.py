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
    # --- STUB: replace with real DB implementation ---
    raise NotImplementedError("store_metadata() is not yet implemented.")
