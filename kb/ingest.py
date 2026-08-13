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

# How far back from a nominal window edge we will look for a clean break before
# giving up and cutting where the word count says.
_SNAP_LOOKBACK = 30

# Trailing markup to ignore when deciding whether a token closes a sentence, so
# that "seven words.**" and 'the end."' still count.
_TRAILING_MARKUP = "\"')]}*_`”’"

# An ordered-list marker ("2." or "2)") ends in a period but opens an item
# rather than closing a sentence. Snapping there would strand the number from
# the text it numbers.
_LIST_MARKER_RE = re.compile(r"\A\d+[.)]\Z")


def _body_offset(content: str) -> int:
    """Char offset where prose begins, skipping any leading YAML frontmatter.

    Frontmatter is identity, not content: embedding ``url`` and
    ``kb-clipped: true`` spends window on every clip and turns those keys into
    terms that match every clipped note.

    Returns 0 when the document does not open with a *closed* frontmatter
    block, so a leading ``---`` used as a thematic break keeps its content.
    """
    first_nl = content.find("\n")
    if first_nl == -1 or content[:first_nl].strip() != "---":
        return 0
    pos = first_nl + 1
    while pos < len(content):
        nl = content.find("\n", pos)
        line_end = len(content) if nl == -1 else nl
        if content[pos:line_end].strip() in ("---", "..."):
            return len(content) if nl == -1 else nl + 1
        pos = line_end + 1
    # Never closed, so this was not frontmatter after all.
    return 0


def _ends_sentence(token: str) -> bool:
    """True when *token* is the last word of a sentence."""
    if _LIST_MARKER_RE.match(token):
        return False
    return token.rstrip(_TRAILING_MARKUP).endswith(
        (".", "!", "?", "。", "！", "？", "…")
    )


def _snap_end(
    content: str, spans: list[tuple[int, int]], start: int, nominal: int
) -> int:
    """Pull a window's exclusive end back to the nearest clean break.

    Looks backward from *nominal* for a line break or a sentence ending, within
    :data:`_SNAP_LOOKBACK` words. Falls back to *nominal* when the text offers
    no boundary — long unbroken prose still gets chunked.
    """
    floor = max(start + 1, nominal - _SNAP_LOOKBACK)
    for end in range(nominal, floor - 1, -1):
        last_start, last_end = spans[end - 1]
        # A newline before the next word ends a line, list item, or paragraph.
        if "\n" in content[last_end : spans[end][0]]:
            return end
        if _ends_sentence(content[last_start:last_end]):
            return end
    return nominal


def _chunk_document(content: str) -> list[tuple[str, int]]:
    """Split ``content`` into overlapping word-windows.

    Returns a list of ``(chunk_text, start_line)`` tuples. ``chunk_text`` is
    sliced from the original string (formatting preserved); ``start_line`` is
    the 1-based line number where the chunk begins, counted in the *whole*
    document so citations survive the frontmatter skip. A document shorter than
    one window yields a single chunk covering all of it.

    Windows end on a sentence or line boundary where one is close by: a chunk
    cut mid-sentence splits a term from the text defining it, which costs the
    answer layer its ability to cite either one precisely.
    """
    offset = _body_offset(content)
    spans = [
        (m.start(), m.end())
        for m in _WORD_RE.finditer(content)
        if m.start() >= offset
    ]
    if not spans:
        return []

    chunks: list[tuple[str, int]] = []
    n = len(spans)
    i = 0
    while i < n:
        nominal = i + _CHUNK_WORDS
        # The final window reaches the end; no snapping and no tiny duplicate tail.
        end = n if nominal >= n else _snap_end(content, spans, i, nominal)

        start_char = spans[i][0]
        end_char = spans[end - 1][1]
        chunk_text = content[start_char:end_char]
        start_line = content.count("\n", 0, start_char) + 1
        chunks.append((chunk_text, start_line))

        if end >= n:
            break
        # Overlap is measured back from where this chunk actually ended.
        i = max(end - _CHUNK_OVERLAP, i + 1)

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

    model = embedding.current_model_name()

    if not rebuild:
        try:
            old_vectors, old_metas = store.load(index_dir)
            # An index built with a different embedding model cannot have its
            # vectors reused -- they live in a different space. Drop it and
            # re-embed everything under the current model.
            old_model = (
                old_metas[0].get("embed_model", embedding.LEGACY_MODEL)
                if old_metas
                else model
            )
            if old_model != model:
                old_vectors = None
                old_by_key = {}
            else:
                for i, m in enumerate(old_metas):
                    old_by_key.setdefault(
                        m.get("key", m.get("path", "")), []
                    ).append((i, m))
        except Exception:
            # Missing or corrupt index → start fresh, no crash.
            old_vectors = None
            old_by_key = {}

    # Keys visible in this walk (used to detect removals).
    candidate_keys = source.candidate_keys()

    # A missing key is a deletion only if it belongs to this source's scope;
    # keys owned by other sources sharing the index are preserved below.
    # getattr fallback keeps third-party Source implementations (no owns_key)
    # on the old replace-everything behavior.
    owns_key = getattr(source, "owns_key", lambda _k: True)
    removed_count = sum(
        1 for k in old_by_key if k not in candidate_keys and owns_key(k)
    )

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
                "embed_model": model,
            }
            chunk_sources.append(("new", len(new_texts)))
            new_texts.append(chunk_text)
            new_metas.append(meta)
            final_metas.append(meta)

    # --- Preserve chunks owned by other sources sharing this index ---
    # Deterministic order: keys sorted, chunks in stored order within a key.
    for key in sorted(old_by_key):
        if key in candidate_keys or owns_key(key):
            continue
        for old_idx, meta in old_by_key[key]:
            final_metas.append(meta)
            chunk_sources.append(("reuse", old_idx))

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
