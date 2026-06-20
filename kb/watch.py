"""Pure-stdlib polling watcher that auto-reingests registered file sources.

This module is the single-process counterpart to the ``kb`` CLI's manual
ingest: it watches every ``"files"`` source in the registry and re-runs the
(already incremental) ingest whenever the on-disk picture changes.

Project red line -- *pure Python, no native-compiled dependencies*.  Rather than
reach for the ``watchdog`` package (which carries platform-specific compiled
extensions), this watcher polls with :func:`os.walk` + ``stat`` and compares a
cheap ``path -> (mtime, size)`` snapshot between cycles.  A difference triggers
a re-ingest; the heavy lifting (vector reuse for unchanged files) is delegated
to :func:`kb.ingest.ingest`, so a no-op change is inexpensive.

Design split
------------
* :func:`_snapshot`  -- one cheap stat-pass over a folder, defensive on errors.
* :func:`run_once`   -- exactly one watch cycle (snapshot, diff, maybe ingest);
                        the testable, loop-free entrypoint.
* :func:`watch`      -- the long-running CLI loop built on :func:`run_once`.

Every per-folder operation is wrapped so that an unreadable or vanished source
is skipped, never fatal -- mirroring the defensive posture of
:mod:`kb.config` and :class:`kb.source.FileSource`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from . import config
from .ingest import ingest as ingest_fn
from .source import FileSource


def _snapshot(folder: str) -> dict[str, tuple[float, int]]:
    """Return a ``path -> (mtime, size)`` map for *folder*'s candidate files.

    Walks *folder* exactly as :class:`~kb.source.FileSource` does (reusing its
    ``candidate_keys`` so the suffix / ignore-dir / size policy stays in one
    place), then ``stat``\ s each path for its ``(mtime, size)`` change token.
    Kept cheap: no content is read.

    Defensive by design -- a missing or unreadable folder yields ``{}`` rather
    than raising, and a file that disappears mid-scan is silently skipped.
    """
    snapshot: dict[str, tuple[float, int]] = {}
    try:
        keys = FileSource(folder).candidate_keys()
    except Exception:
        return {}

    for path in keys:
        try:
            stat = os.stat(path)
        except Exception:
            continue  # vanished mid-scan or unreadable -- skip
        snapshot[path] = (stat.st_mtime, stat.st_size)

    return snapshot


def _files_folders() -> list[str]:
    """Return the normalized paths of every registered ``"files"`` source.

    Bookmark sources are intentionally ignored by the watcher at this stage.
    """
    return [
        entry["path"]
        for entry in config.load_sources()
        if entry.get("kind") == "files" and entry.get("path")
    ]


def run_once(
    index_dir: Optional[str] = None,
    state: Optional[dict] = None,
) -> tuple[bool, dict]:
    """Run exactly one watch cycle; return ``(changed, new_state)``.

    This is the loop-free entrypoint so callers (and tests) can drive a single
    pass deterministically.

    The cycle:

    1. Resolve *index_dir* to :func:`kb.config.default_index_dir` when ``None``.
    2. Build a combined ``path -> (mtime, size)`` snapshot across every
       registered ``"files"`` source (bookmarks are ignored here).
    3. Compare to *state* (a snapshot of the same shape).  ``None``/empty is the
       *first run* -- when there are registered files it counts as *changed*.
    4. On any difference (added, removed, or changed file) re-ingest each
       registered folder via the incremental :func:`kb.ingest.ingest` and return
       ``(True, snapshot)``.  Otherwise do no ingest and return
       ``(False, snapshot)``.

    Per-folder ingest is wrapped so one unreadable source cannot abort the pass.
    """
    if index_dir is None:
        index_dir = config.default_index_dir()

    folders = _files_folders()

    snapshot: dict[str, tuple[float, int]] = {}
    for folder in folders:
        snapshot.update(_snapshot(folder))

    previous = state or {}
    # First-run semantics: empty prior state + present files == changed.
    changed = snapshot != previous

    if changed:
        for folder in folders:
            try:
                ingest_fn(folder, index_dir=index_dir)
            except Exception as exc:
                # An unreadable / mid-delete source must not abort the cycle,
                # but surface the reason so a real breakage isn't masked.
                print(f"kb watch: skipping {folder}: {exc}", file=sys.stderr)
                continue

    return changed, snapshot


def watch(index_dir: Optional[str] = None, interval: float = 3.0) -> None:
    """Poll registered ``"files"`` sources forever, re-ingesting on any change.

    The long-running CLI loop.  Starts from an empty state, then repeatedly
    calls :func:`run_once`, sleeping *interval* seconds between cycles.  A
    re-ingest is announced only when something actually changed.

    A ``KeyboardInterrupt`` (Ctrl-C) ends the loop cleanly with a short
    "stopped" line.  Each cycle is delegated to :func:`run_once`, which already
    isolates per-folder failures, so an unreadable source is skipped rather than
    fatal.
    """
    if index_dir is None:
        index_dir = config.default_index_dir()

    folders = _files_folders()
    print(
        f"Watching {len(folders)} file source(s); "
        f"polling every {interval}s. Ctrl-C to stop."
    )

    state: Optional[dict] = None
    try:
        while True:
            try:
                changed, state = run_once(index_dir, state)
                if changed:
                    print("Detected changes -- reingested.")
            except Exception:
                # Belt-and-suspenders: a cycle-level failure must not kill the
                # watcher; the next poll retries.
                pass
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Watch stopped.")
        return
