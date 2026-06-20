r"""Managed per-user configuration: home directory, index location, and a JSON
source registry.

This module is the foundation that lets the ``kb`` CLI become zero-config.  It
owns three pieces of policy, all derived from a single managed *home* directory:

* :func:`kb_home`            -- the per-user config/data root, platform-aware.
* :func:`default_index_dir`  -- where the vector index lives by default.
* a small JSON *source registry* (``sources.json`` under the home) recording
  which origins the user has registered for ingest.

Home-directory resolution (queryable, never side-effecting)
-----------------------------------------------------------
``kb_home`` is pure: it computes a path and returns it without ever creating a
directory.  Resolution order:

1. ``$KB_HOME`` if set and non-empty -- the test/override hook, so tests and
   callers can redirect the entire managed home away from the real user config.
2. Windows  (``os.name == "nt"``):   ``%APPDATA%\kb``  (``~/.kb`` if APPDATA
   is missing).
3. macOS    (``sys.platform == "darwin"``): ``~/Library/Application Support/kb``.
4. Otherwise (Linux/other): ``$XDG_DATA_HOME/kb`` if set, else
   ``~/.local/share/kb``.

Directory creation is deferred to the helpers that actually write
(:func:`add_source` / :func:`remove_source`); read paths stay side-effect free.

Source registry shape
----------------------
``sources.json`` is a JSON list of objects::

    [{"kind": "files", "path": "/abs/notes"},
     {"kind": "bookmarks", "path": "/abs/Bookmarks"}]

``kind`` is restricted to ``"files"`` or ``"bookmarks"`` (the two concrete
:class:`~kb.source.Source` adapters).  Every read is defensive: a missing,
unreadable, non-JSON, or non-list registry yields ``[]`` rather than raising,
mirroring the posture of :class:`~kb.source.FileSource` and the ingest loop.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allowed values for a registered source's "kind" field.
_VALID_KINDS = {"files", "bookmarks"}


def kb_home() -> Path:
    r"""Return the managed per-user home directory for ``kb`` (never created).

    ``$KB_HOME`` wins when set and non-empty (the test/override hook).  Else a
    platform default is chosen: ``%APPDATA%\kb`` on Windows, ``~/Library/
    Application Support/kb`` on macOS, and ``$XDG_DATA_HOME/kb`` (falling back to
    ``~/.local/share/kb``) elsewhere.  This function is pure and queryable -- it
    does not create the directory.
    """
    override = os.environ.get("KB_HOME")
    if override:
        return Path(override)

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "kb"
        return Path.home() / ".kb"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kb"

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "kb"
    return Path.home() / ".local" / "share" / "kb"


def default_index_dir() -> str:
    """Return the default on-disk index location, ``<kb_home>/index`` as a str."""
    return str(kb_home() / "index")


def sources_path() -> Path:
    """Return the path to the JSON source registry, ``<kb_home>/sources.json``."""
    return kb_home() / "sources.json"


def load_sources() -> list[dict]:
    """Return the registered sources as a list of dicts; ``[]`` on any problem.

    Defensive by design: a missing, unreadable, non-JSON, or non-list registry
    returns an empty list rather than raising, mirroring the codebase posture.
    """
    try:
        raw = sources_path().read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def _normalize(path: str) -> str:
    """Normalize a filesystem path for storage and dedup comparisons."""
    return str(Path(path))


def _write_sources(sources: list[dict]) -> None:
    """Create ``kb_home`` if needed and write *sources* as pretty JSON."""
    home = kb_home()
    home.mkdir(parents=True, exist_ok=True)
    sources_path().write_text(
        json.dumps(sources, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_source(kind: str, path: str) -> list[dict]:
    """Register a source of *kind* at *path*, dedup, persist, and return the list.

    ``kind`` must be ``"files"`` or ``"bookmarks"`` (raises ``ValueError``
    otherwise).  The path is normalized; an entry matching an existing
    ``(kind, path)`` is not duplicated.  ``kb_home`` is created on write.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"invalid source kind {kind!r}; expected one of {sorted(_VALID_KINDS)}"
        )
    norm = _normalize(path)
    sources = load_sources()
    for entry in sources:
        if entry.get("kind") == kind and entry.get("path") == norm:
            return sources  # already registered -- no duplicate
    sources.append({"kind": kind, "path": norm})
    _write_sources(sources)
    return sources


def remove_source(path: str) -> list[dict]:
    """Remove every registered source whose path matches *path*; return the list.

    Comparison is by normalized path (kind-agnostic).  Absent entries are a
    no-op.  The updated list is written back and returned.
    """
    norm = _normalize(path)
    sources = load_sources()
    kept = [entry for entry in sources if entry.get("path") != norm]
    if len(kept) != len(sources):
        _write_sources(kept)
    return kept
