"""Ingest pipeline: walk a directory, embed notes, and persist an index."""

from __future__ import annotations

import re

import numpy

from . import embedding, store
from .source import FileSource

# Default location for the on-disk index. Must match query.py and the CLI.
DEFAULT_INDEX_DIR = ".kb_index"

# Chunking parameters: ~200 words per chunk with a 40-word overlap so that a
# passage straddling a boundary still lands wholly inside at least one chunk.
_CHUNK_WORDS = 200
_CHUNK_OVERLAP = 40

# A "word" is any run of non-whitespace characters.
_WORD_RE = re.compile(r"\S+")


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


def ingest(
    source_dir: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> int:
    """Index every supported source file under ``source_dir`` into ``index_dir``.

    In incremental mode (``rebuild=False``) files whose change token
    (``(mtime, size)`` for local files) is unchanged since the last run have
    their existing vectors reused; only new or modified files are re-embedded.
    Files that no longer exist on disk are dropped from the index.

    Args:
        source_dir: Directory to walk recursively for notes and code.
        index_dir: Directory the vector index is written into.
        rebuild: When True, ignore any existing index and re-embed everything.

    Returns:
        The number of chunks indexed (a single file may produce several).
    """
    source = FileSource(source_dir)

    # --- Load existing index for incremental diffing ---
    old_vectors: numpy.ndarray | None = None
    # key -> list of (row_index_in_old_vectors, meta_dict)
    old_by_key: dict[str, list[tuple[int, dict]]] = {}

    if not rebuild:
        try:
            old_vectors, old_metas = store.load(index_dir)
            for i, m in enumerate(old_metas):
                old_by_key.setdefault(m.get("path", ""), []).append((i, m))
        except Exception:
            # Missing or corrupt index → start fresh, no crash.
            old_vectors = None
            old_by_key = {}

    # Keys visible in this walk (used to detect removals).
    candidate_keys = source.candidate_keys()

    # Keys in the old index that are no longer present on disk.
    removed_count = sum(1 for k in old_by_key if k not in candidate_keys)

    # --- Classify each candidate and build output lists in walk order ---
    # chunk_sources[i] is either ('reuse', old_row_idx) or ('new', new_batch_idx).
    final_metas: list[dict] = []
    chunk_sources: list[tuple[str, int]] = []

    new_texts: list[str] = []   # texts queued for embedding
    new_metas: list[dict] = []  # parallel to new_texts

    n_files_reembedded = 0
    n_files_unchanged = 0

    for doc in source.documents():
        old_chunks = old_by_key.get(doc.key, [])

        # Reconstruct the stored change token from old meta for comparison.
        # If 'size' is absent the stored token is treated as missing (safe
        # full re-embed; no crash on old indexes that predate the size field).
        if old_chunks and "size" in old_chunks[0][1]:
            stored_token: tuple | None = (
                old_chunks[0][1]["mtime"],
                old_chunks[0][1]["size"],
            )
        else:
            stored_token = None

        is_unchanged = bool(old_chunks) and stored_token == doc.change_token

        if is_unchanged:
            n_files_unchanged += 1
            for old_idx, meta in old_chunks:
                final_metas.append(meta)
                chunk_sources.append(("reuse", old_idx))
            continue

        # New or changed: load content and queue for embedding.
        content = doc.content()
        if content is None:
            continue

        n_files_reembedded += 1

        for chunk_text, start_line in _chunk_document(content):
            meta = {
                "path": doc.path,
                "filename": doc.filename,
                "chunk_text": chunk_text,
                "start_line": start_line,
                "mtime": doc.mtime,
                "size": doc.size,
                "kind": doc.kind,
            }
            chunk_sources.append(("new", len(new_texts)))
            new_texts.append(chunk_text)
            new_metas.append(meta)
            final_metas.append(meta)

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
