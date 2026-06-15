"""Ingest pipeline: walk a directory, embed notes, and persist an index."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy

from . import embedding, store

# Default location for the on-disk index. Must match query.py and the CLI.
DEFAULT_INDEX_DIR = ".kb_index"

# Extensions we treat as ingestible text/code (matched case-insensitively).
_TEXT_SUFFIXES = {
    ".md", ".txt", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".rb", ".c", ".cpp", ".h", ".sh",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css",
}

# Suffixes that indicate source code (everything else → 'note').
_CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".rb", ".c", ".cpp", ".h", ".sh",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css",
}

# Directories that are never worth indexing.
_IGNORE_DIRS = {
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "dist", "build", ".next", "target", ".idea", ".vscode", "coverage",
    ".mypy_cache", ".pytest_cache", ".tox", ".kb_index",
}

# Files larger than this are skipped (avoids embedding megabyte blobs).
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB

# Chunking parameters: ~200 words per chunk with a 40-word overlap so that a
# passage straddling a boundary still lands wholly inside at least one chunk.
_CHUNK_WORDS = 200
_CHUNK_OVERLAP = 40

# A "word" is any run of non-whitespace characters.
_WORD_RE = re.compile(r"\S+")


def _kind(suffix: str) -> str:
    return "code" if suffix.lower() in _CODE_SUFFIXES else "note"


def _chunk_document(content: str) -> list[tuple[str, int]]:
    """Split ``content`` into overlapping word-windows.

    Returns a list of ``(chunk_text, start_line)`` tuples. ``chunk_text`` is
    sliced from the original string (formatting preserved); ``start_line`` is
    the 1-based line number where the chunk begins. A document shorter than one
    window yields a single chunk covering all of it.
    """
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(content)]
    if not spans:
        return []

    step = max(_CHUNK_WORDS - _CHUNK_OVERLAP, 1)
    chunks: list[tuple[str, int]] = []
    n = len(spans)
    i = 0
    while i < n:
        window = spans[i : i + _CHUNK_WORDS]
        start_char = window[0][0]
        end_char = window[-1][1]
        chunk_text = content[start_char:end_char]
        start_line = content.count("\n", 0, start_char) + 1
        chunks.append((chunk_text, start_line))

        # The final window reaches the end; stop to avoid a tiny duplicate tail.
        if i + _CHUNK_WORDS >= n:
            break
        i += step

    return chunks


def _walk_files(root: Path) -> list[Path]:
    """Collect candidate files using os.walk, pruning noise directories."""
    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in-place so os.walk never descends into ignored dirs.
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in _TEXT_SUFFIXES:
                collected.append(p)
    return sorted(collected)


def ingest(
    source_dir: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> int:
    """Index every supported source file under ``source_dir`` into ``index_dir``.

    In incremental mode (``rebuild=False``) files whose ``(mtime, size)`` are
    unchanged since the last run have their existing vectors reused; only
    new or modified files are re-embedded.  Files that no longer exist on disk
    are dropped from the index.

    Args:
        source_dir: Directory to walk recursively for notes and code.
        index_dir: Directory the vector index is written into.
        rebuild: When True, ignore any existing index and re-embed everything.

    Returns:
        The number of chunks indexed (a single file may produce several).
    """
    root = Path(source_dir)
    paths = _walk_files(root)

    # --- Load existing index for incremental diffing ---
    old_vectors: numpy.ndarray | None = None
    # path -> list of (row_index_in_old_vectors, meta_dict)
    old_by_path: dict[str, list[tuple[int, dict]]] = {}

    if not rebuild:
        try:
            old_vectors, old_metas = store.load(index_dir)
            for i, m in enumerate(old_metas):
                old_by_path.setdefault(m.get("path", ""), []).append((i, m))
        except Exception:
            # Missing or corrupt index → start fresh, no crash.
            old_vectors = None
            old_by_path = {}

    # Paths visible on disk in this walk (used to detect removals).
    candidate_path_strs = {str(p) for p in paths}

    # Files in the old index that are no longer on disk.
    removed_count = sum(1 for p in old_by_path if p not in candidate_path_strs)

    # --- Classify each candidate file and build output lists in walk order ---
    # chunk_sources[i] is either ('reuse', old_row_idx) or ('new', new_batch_idx).
    final_metas: list[dict] = []
    chunk_sources: list[tuple[str, int]] = []

    new_texts: list[str] = []   # texts queued for embedding
    new_metas: list[dict] = []  # parallel to new_texts

    n_files_reembedded = 0
    n_files_unchanged = 0

    for p in paths:
        path_str = str(p)
        try:
            stat = os.stat(p)
            if stat.st_size > _MAX_FILE_BYTES:
                continue

            file_mtime = stat.st_mtime
            file_size = stat.st_size

            old_chunks = old_by_path.get(path_str, [])

            # Decide reuse vs. re-embed.
            # If any old chunk is missing 'size' the entire file is treated as
            # changed (safe full re-embed; no crash on old indexes).
            is_unchanged = (
                bool(old_chunks)
                and "size" in old_chunks[0][1]
                and old_chunks[0][1]["mtime"] == file_mtime
                and old_chunks[0][1]["size"] == file_size
            )

            if is_unchanged:
                n_files_unchanged += 1
                for old_idx, meta in old_chunks:
                    final_metas.append(meta)
                    chunk_sources.append(("reuse", old_idx))
                continue

            # New or changed: read, chunk, and queue for embedding.
            raw = p.read_bytes()
            # Decode by BOM when present; UTF-16 legitimately contains NUL
            # bytes, so the NUL heuristic must only apply AFTER ruling out a
            # text BOM.
            if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
                try:
                    content = raw.decode("utf-16")
                except (UnicodeDecodeError, ValueError):
                    continue  # corrupt UTF-16 -> skip
            elif raw.startswith(b"\xef\xbb\xbf"):
                content = raw.decode("utf-8-sig", errors="replace")
            elif b"\x00" in raw:
                continue  # real binary
            else:
                content = raw.decode("utf-8", errors="replace")
            if not content.strip():
                continue

            kind = _kind(p.suffix)
            n_files_reembedded += 1

            for chunk_text, start_line in _chunk_document(content):
                meta = {
                    "path": path_str,
                    "filename": p.name,
                    "chunk_text": chunk_text,
                    "start_line": start_line,
                    "mtime": file_mtime,
                    "size": file_size,
                    "kind": kind,
                }
                chunk_sources.append(("new", len(new_texts)))
                new_texts.append(chunk_text)
                new_metas.append(meta)
                final_metas.append(meta)

        except Exception:
            # One bad file must not abort the whole ingest run.
            continue

    if not final_metas:
        print(f"No ingestible files found in '{source_dir}'. Nothing was indexed.")
        return 0

    # --- Embed only the changed/new texts ---
    new_vectors: numpy.ndarray | None = None
    if new_texts:
        new_vectors = embedding.encode(new_texts)

    # --- Assemble final vector matrix aligned with final_metas ---
    ref = new_vectors if new_vectors is not None else old_vectors
    dim = ref.shape[1]  # type: ignore[union-attr]
    dtype = ref.dtype   # type: ignore[union-attr]

    final_vectors = numpy.empty((len(final_metas), dim), dtype=dtype)
    for slot, (source_type, source_idx) in enumerate(chunk_sources):
        if source_type == "reuse":
            final_vectors[slot] = old_vectors[source_idx]  # type: ignore[index]
        else:
            final_vectors[slot] = new_vectors[source_idx]  # type: ignore[index]

    store.save(final_vectors, final_metas, index_dir)

    total = len(final_metas)
    print(
        f"Indexed {total} chunks "
        f"({n_files_reembedded} files re-embedded, "
        f"{n_files_unchanged} unchanged, "
        f"{removed_count} removed)."
    )
    return total
