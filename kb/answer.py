"""Answer synthesis: turn ranked chunks from query() into a grounded, cited answer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .llm import ClaudeProvider, LLMProvider
from .query import DEFAULT_INDEX_DIR, query

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Frozen system prompt — constant across all calls so the LLM cache can reuse
# the encoded representation after the first request.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledge-base assistant. "
    "Answer the user's question using ONLY the numbered source passages supplied "
    "in the user turn. "
    "Cite every claim inline with the passage number(s) in square brackets, "
    "e.g. [1] or [2][4]. "
    "If the passages do not contain enough information to answer the question, "
    "say so clearly — do not invent facts. "
    "Be concise and direct."
)

# ---------------------------------------------------------------------------
# Empty-retrieval sentinel — returned verbatim when the index has no matches,
# so the caller never sees a blank answer and the model is never invoked.
# ---------------------------------------------------------------------------

_NO_CONTEXT_ANSWER = (
    "No relevant content found in the knowledge base for this question."
)


def answer(
    question: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    k: int = 5,
    provider: LLMProvider | None = None,
    since: str | None = None,
    kind: str | None = None,
    hybrid: bool = False,
) -> dict:
    """Return a grounded answer with citations for *question*.

    Parameters
    ----------
    question:
        The natural-language query to answer.
    index_dir:
        Directory containing the vector index (forwarded to :func:`query`).
    k:
        Number of top chunks to retrieve and include in the context window.
    provider:
        An :class:`~kb.llm.LLMProvider` instance.  Defaults to
        :class:`~kb.llm.ClaudeProvider` when ``None``.
    since:
        Optional recency window forwarded to :func:`query` (e.g. ``"7d"``).
    kind:
        Optional kind filter forwarded to :func:`query` (``"note"`` / ``"code"``).
    hybrid:
        When ``True``, use RRF-fused semantic + BM25 retrieval.

    Returns
    -------
    dict with keys:

    ``"answer"``
        The model's text response, with inline citation markers.
    ``"citations"``
        A list of dicts ``{"n", "filename", "path", "start_line"}`` — one per
        retrieved chunk, numbered to match the ``[N]`` markers in the answer.
    ``"used_chunks"``
        The raw chunk dicts returned by :func:`query`, in the same order as
        ``"citations"``.
    """
    chunks = query(
        question,
        index_dir=index_dir,
        k=k,
        since=since,
        kind=kind,
        hybrid=hybrid,
    )

    # Empty-retrieval fast path — no model call, no citations.
    if not chunks:
        return {
            "answer": _NO_CONTEXT_ANSWER,
            "citations": [],
            "used_chunks": [],
        }

    # Build a provenance-tagged context block for the user turn.
    # Each chunk gets a "[N] filename · line M" header so the model can cite it.
    context_parts: list[str] = []
    for n, chunk in enumerate(chunks, 1):
        header = f"[{n}] {chunk['filename']} · line {chunk['start_line']}"
        context_parts.append(f"{header}\n{chunk['excerpt']}")
    context = "\n\n".join(context_parts)

    if provider is None:
        provider = ClaudeProvider()

    answer_text = provider.complete(_SYSTEM_PROMPT, context, question)

    citations = [
        {
            "n": n,
            "filename": chunk["filename"],
            "path": chunk["path"],
            "start_line": chunk["start_line"],
        }
        for n, chunk in enumerate(chunks, 1)
    ]

    return {
        "answer": answer_text,
        "citations": citations,
        "used_chunks": chunks,
    }
