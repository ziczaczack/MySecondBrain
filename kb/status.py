"""Index statistics: summarize an existing on-disk index without re-embedding."""

from __future__ import annotations

import os
import time

from . import store

# Default location for the on-disk index. Must match ingest.py and the CLI.
DEFAULT_INDEX_DIR = ".kb_index"


def status(index_dir: str = DEFAULT_INDEX_DIR) -> dict:
    """Summarize the index in ``index_dir`` without embedding anything.

    Loads the persisted vectors and metadata and reports counts, a per-kind
    breakdown, the on-disk size, and when the index was last written. This is a
    read-only operation: nothing is re-embedded or rewritten.

    Args:
        index_dir: Directory containing the vector index.

    Returns:
        A dict. When no index exists, ``{"exists": False, "index_dir": ...}``.
        Otherwise a dict with ``exists``, ``index_dir``, ``files`` (distinct
        source paths), ``chunks`` (total vectors), ``kinds`` (kind -> count),
        ``index_bytes`` (size of the index files on disk), ``last_ingest``
        (Unix float mtime of meta.json, or ``None``), and ``last_ingest_date``
        (``"YYYY-MM-DD HH:MM"`` local time, or ``""``).
    """
    try:
        _vectors, metas = store.load(index_dir)
    except FileNotFoundError:
        return {"exists": False, "index_dir": index_dir}

    # Distinct source files and a per-kind tally over all chunks.
    paths: set[str] = set()
    kinds: dict[str, int] = {}
    for m in metas:
        paths.add(m.get("path", ""))
        kind = m.get("kind", "note")
        kinds[kind] = kinds.get(kind, 0) + 1

    # Size on disk of the two index files. Use store's filename constants so we
    # stay in sync with how the store names them; tolerate a missing file.
    vectors_path = os.path.join(index_dir, store._VECTORS_FILE)
    meta_path = os.path.join(index_dir, store._META_FILE)
    index_bytes = 0
    for fp in (vectors_path, meta_path):
        try:
            index_bytes += os.path.getsize(fp)
        except OSError:
            # Missing or unreadable file just contributes nothing.
            pass

    # The metadata file's mtime stands in for "when the index was last built".
    try:
        last_ingest: float | None = os.path.getmtime(meta_path)
    except OSError:
        last_ingest = None
    last_ingest_date = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ingest))
        if last_ingest is not None
        else ""
    )

    return {
        "exists": True,
        "index_dir": index_dir,
        "files": len(paths),
        "chunks": len(metas),
        "kinds": kinds,
        "index_bytes": index_bytes,
        "last_ingest": last_ingest,
        "last_ingest_date": last_ingest_date,
    }
