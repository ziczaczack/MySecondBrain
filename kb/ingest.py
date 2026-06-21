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

# CJK Unicode ranges treated as single-character tokens so that Chinese,
# Japanese, and Korean text chunks at character granularity rather than
# swallowing whole sentences into one "word".
_CJK_CHARS = (
    r"一-鿿"   # CJK Unified Ideographs
    r"㐀-䶿"   # CJK Extension A
    r"豈-﫿"   # CJK Compatibility Ideographs
    r"぀-ゟ"   # Hiragana
    r"゠-ヿ"   # Katakana
    r"가-힯"   # Hangul Syllables
)
# Each CJK character is one token; a run of non-whitespace non-CJK chars is
# one token (identical to the previous \S+ behaviour for pure-ASCII text).
_WORD_RE = re.compile(rf"[{_CJK_CHARS}]|[^\s{_CJK_CHARS}]+")


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
    return _ingest_from_source(
        source, index_dir=index_dir, rebuild=rebuild, label=source_dir
    )


def _ingest_from_source(
    source,
    index_dir: str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
    label: str = "the source",
) -> int:
    """Index every Document produced by ``source`` into ``index_dir``.

    This is the source-agnostic core of the ingest pipeline: it drives any
    object satisfying the :class:`~kb.source.Source` protocol through the same
    incremental diff, classification, embedding, and persistence steps.

    Args:
        source: Any object satisfying the Source protocol.
        index_dir: Directory the vector index is written into.
        rebuild: When True, ignore any existing index and re-embed everything.
        label: Human-readable origin name for the empty-result message.

    Returns:
        The number of chunks indexed (a single document may produce several).
    """
    # --- Load existing index for incremental diffing ---
    old_vectors: numpy.ndarray | None = None
    # key -> list of (row_index_in_old_vectors, meta_dict)
    old_by_key: dict[str, list[tuple[int, dict]]] = {}

    if not rebuild:
        try:
            old_vectors, old_metas = store.load(index_dir)
            for i, m in enumerate(old_metas):
                old_by_key.setdefault(m.get("key", m.get("path", "")), []).append((i, m))
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
        # Source-agnostic: prefer the persisted change_token (any source), fall
        # back to (mtime, size) for legacy pre-Stage-4 indexes, else treat as
        # missing (safe full re-embed; no crash on old indexes).
        stored_token: tuple | None = None
        if old_chunks:
            stored = old_chunks[0][1]
            if "change_token" in stored:
                stored_token = tuple(stored["change_token"])
            elif "size" in stored:                       # legacy index (pre-Stage-4)
                stored_token = (stored["mtime"], stored["size"])
            else:
                stored_token = None

        # doc.change_token coerced to tuple: the stored side round-trips
        # through json as a list, so compare tuple-to-tuple.
        is_unchanged = bool(old_chunks) and stored_token == tuple(doc.change_token)

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
                "key": doc.key,
                "path": doc.path,
                "filename": doc.filename,
                "chunk_text": chunk_text,
                "start_line": start_line,
                "mtime": doc.mtime,
                "size": doc.size,
                "kind": doc.kind,
                "change_token": list(doc.change_token),
            }
            chunk_sources.append(("new", len(new_texts)))
            new_texts.append(chunk_text)
            new_metas.append(meta)
            final_metas.append(meta)

    if not final_metas:
        print(f"No ingestible files found in '{label}'. Nothing was indexed.")
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
