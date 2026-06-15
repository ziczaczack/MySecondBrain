"""Query pipeline: embed a question and search the persisted index."""

from __future__ import annotations

import re
import time
from datetime import datetime

from . import embedding, store

# Default location for the on-disk index. Must match ingest.py and the CLI.
DEFAULT_INDEX_DIR = ".kb_index"

# How much of a matched chunk to surface as the excerpt.
_EXCERPT_MAX_LEN = 240

# Seconds in a day, used when interpreting a relative "<N>d" since window.
_SECONDS_PER_DAY = 86400.0


class IncompatibleIndexError(RuntimeError):
    """Raised when the on-disk index predates chunk-level metadata."""


def _parse_since(since, now=None):
    """Parse a ``since`` window into a Unix-float cutoff timestamp.

    Args:
        since: The window to parse. ``None`` or ``""`` means no filtering;
            ``"<N>d"`` (e.g. ``"7d"``) means the last ``N`` days; an absolute
            ``"YYYY-MM-DD"`` date means that calendar day at local midnight.
        now: Reference current time as a Unix float, for testability.
            Defaults to ``time.time()``.

    Returns:
        The cutoff as a Unix float, or ``None`` when no filtering applies.

    Raises:
        ValueError: If ``since`` is neither ``"<N>d"`` nor ``"YYYY-MM-DD"``.
    """
    if not since:
        return None
    now_ts = time.time() if now is None else now
    s = since.strip()
    # Relative window: "<N>d" counts back N whole days from ``now``.
    if s.endswith("d") and s[:-1].isdigit():
        return now_ts - int(s[:-1]) * _SECONDS_PER_DAY
    # Absolute date: that calendar day at 00:00 local time.
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Unrecognized --since value: {since!r}. "
            "Use '<N>d' or 'YYYY-MM-DD'."
        )
    return dt.timestamp()


def _make_excerpt(chunk_text: str) -> str:
    """Collapse whitespace and truncate a chunk into a one-line excerpt."""
    collapsed = " ".join(chunk_text.split())
    if len(collapsed) <= _EXCERPT_MAX_LEN:
        return collapsed
    return collapsed[:_EXCERPT_MAX_LEN].rstrip() + "…"


def _matched_terms(question: str, excerpt: str) -> list[str]:
    """Return the distinct query terms (>=2 chars) that appear in the excerpt, case-insensitive, in query order."""
    terms = []
    seen = set()
    low = excerpt.lower()
    for tok in re.findall(r"\w+", question.lower()):
        if len(tok) >= 2 and tok not in seen and tok in low:
            seen.add(tok)
            terms.append(tok)
    return terms


def query(
    question: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    k: int = 5,
    since: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    """Return the top-``k`` chunks most similar to ``question``.

    Args:
        question: The natural-language query.
        index_dir: Directory containing the vector index.
        k: Number of results to return.
        since: Optional recency window restricting results to chunks whose
            source was modified within it: ``"<N>d"`` (last N days) or an
            absolute ``"YYYY-MM-DD"`` date. ``None`` disables filtering.
        kind: Optional kind filter (e.g. ``"note"`` or ``"code"``). When set,
            only chunks whose ``kind`` metadata matches are searched. ``None``
            disables filtering. Chunks from old indexes without a ``kind`` key
            are treated as ``"note"``.

    Returns:
        A list of result dicts ``{"filename", "path", "excerpt", "score",
        "start_line", "mtime", "date", "kind"}`` ordered best-first.
        ``excerpt`` is the matched passage; ``mtime``/``date`` describe the
        source freshness; ``kind`` is ``"note"`` or ``"code"``.

    Raises:
        FileNotFoundError: If no index exists in ``index_dir``.
        IncompatibleIndexError: If the index was built before chunk-level
            metadata existed and must be rebuilt with ``kb ingest``, or if
            ``since`` is requested against an index lacking ``mtime``.
        ValueError: If ``since`` is not a recognized window.
    """
    vectors, metas = store.load(index_dir)

    # Old indexes stored a per-file "summary" instead of chunk metadata. They
    # are not forward-compatible, so ask the user to re-ingest rather than
    # silently returning degraded results.
    if metas and "chunk_text" not in metas[0]:
        raise IncompatibleIndexError(
            f"The index in '{index_dir}' uses an old format without chunk "
            "metadata. Rebuild it with `python -m kb ingest <dir>`."
        )

    # Pre-mtime indexes cannot satisfy a --since window. Only complain when
    # filtering is actually requested so old indexes keep working otherwise.
    if since and metas and "mtime" not in metas[0]:
        raise IncompatibleIndexError(
            f"The index in '{index_dir}' lacks time information (mtime). "
            "Rebuild it with `python -m kb ingest <dir>` to use --since."
        )

    query_vec = embedding.encode_one(question)

    cutoff = _parse_since(since)

    # Build the candidate set by applying since and kind filters together.
    if cutoff is None and kind is None:
        # No filters: rank the full corpus exactly as before.
        hits = store.search(query_vec, vectors, k=k)
    else:
        kept = [
            i
            for i in range(len(metas))
            if (
                cutoff is None
                or (
                    metas[i].get("mtime") is not None
                    and metas[i]["mtime"] >= cutoff
                )
            )
            and (kind is None or metas[i].get("kind", "note") == kind)
        ]
        subset = vectors[kept]
        local_hits = store.search(query_vec, subset, k=k)
        hits = [(kept[local_idx], score) for local_idx, score in local_hits]

    results: list[dict] = []
    for idx, score in hits:
        meta = metas[idx]
        mtime = meta.get("mtime")
        date = (
            time.strftime("%Y-%m-%d", time.localtime(mtime))
            if mtime is not None
            else ""
        )
        excerpt = _make_excerpt(meta.get("chunk_text", ""))
        results.append(
            {
                "filename": meta.get("filename", ""),
                "path": meta.get("path", ""),
                "excerpt": excerpt,
                "start_line": meta.get("start_line", 1),
                "score": score,
                "mtime": mtime,
                "date": date,
                "kind": meta.get("kind", "note"),
                "matched_terms": _matched_terms(question, excerpt),
            }
        )
    return results
