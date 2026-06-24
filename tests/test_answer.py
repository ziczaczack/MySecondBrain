"""Acceptance tests for the answer/RAG synthesis layer (kb.answer).

These tests exercise :func:`kb.answer.answer` in isolation from both the real
vector index and the live Anthropic API:

* retrieval is stubbed by monkeypatching ``kb.answer.query`` to return canned
  chunk dicts shaped like the real :func:`kb.query.query` output, and
* synthesis is stubbed by injecting a ``FakeProvider`` via the ``provider=``
  parameter, which records the (system, context, question) it was called with
  and returns a canned answer string.

So no MiniLM model is loaded (query never runs) and no network/API call is
made (the real ClaudeProvider is never constructed). Conventions mirror
``test_query.py`` / ``test_zeroconfig.py``: pytest + monkeypatch, no fixtures
beyond the built-ins.
"""

import pytest

from kb import answer as answer_mod
from kb.answer import answer, _NO_CONTEXT_ANSWER, _SYSTEM_PROMPT


class FakeProvider:
    """An LLMProvider stand-in that records its call and returns a canned answer.

    ``complete`` satisfies the :class:`kb.llm.LLMProvider` protocol. It stores
    the exact arguments it was handed so tests can assert on the assembled
    context, and flips ``called`` so the empty-retrieval test can prove the
    provider was *never* invoked.
    """

    def __init__(self, canned: str = "FAKE ANSWER [1]"):
        self.canned = canned
        self.called = False
        self.calls: list[dict] = []

    def complete(self, system: str, context: str, question: str) -> str:
        self.called = True
        self.calls.append(
            {"system": system, "context": context, "question": question}
        )
        return self.canned


def _chunk(filename, path, start_line, excerpt, **extra):
    """Build a chunk dict shaped like query()'s output (extra keys allowed)."""
    base = {
        "filename": filename,
        "path": path,
        "start_line": start_line,
        "excerpt": excerpt,
        "score": 0.5,
        "kind": "note",
    }
    base.update(extra)
    return base


# Two canned chunks reused across tests; distinct files, lines, and excerpts.
CHUNKS = [
    _chunk("rust-async.md", "/notes/rust-async.md", 12, "The tokio runtime drives async tasks."),
    _chunk("futures.md", "/notes/futures.md", 88, "A future is an eventual result of work."),
]


def _patch_query(monkeypatch, returned):
    """Monkeypatch kb.answer.query to return *returned*, recording its kwargs."""
    recorded = {}

    def fake_query(question, **kwargs):
        recorded["question"] = question
        recorded["kwargs"] = kwargs
        return returned

    monkeypatch.setattr(answer_mod, "query", fake_query)
    return recorded


def test_context_assembly_tags_each_chunk(monkeypatch):
    """answer() builds a provenance-tagged context: [n] filename header + excerpt, in order."""
    _patch_query(monkeypatch, list(CHUNKS))
    fake = FakeProvider()

    result = answer("what is tokio?", provider=fake)

    assert fake.called, "provider must be called when chunks exist"
    assert len(fake.calls) == 1
    call = fake.calls[0]

    # The frozen system prompt is forwarded verbatim.
    assert call["system"] == _SYSTEM_PROMPT
    # The raw user question is forwarded verbatim.
    assert call["question"] == "what is tokio?"

    context = call["context"]
    # Each chunk appears with its [n] number, filename, line header, and excerpt.
    for n, chunk in enumerate(CHUNKS, 1):
        header = f"[{n}] {chunk['filename']}"
        assert header in context, f"missing provenance header for chunk {n}: {context!r}"
        assert str(chunk["start_line"]) in context, f"missing start_line for chunk {n}: {context!r}"
        assert chunk["excerpt"] in context, f"missing excerpt for chunk {n}: {context!r}"

    # Order is preserved: chunk 1's header precedes chunk 2's header.
    pos1 = context.index("[1] rust-async.md")
    pos2 = context.index("[2] futures.md")
    assert pos1 < pos2, f"chunks not in retrieval order in context: {context!r}"
    # And each excerpt sits after its own header.
    assert context.index("tokio runtime") > pos1
    assert context.index("eventual result") > pos2


def test_citation_mapping_and_used_chunks(monkeypatch):
    """citations are numbered 1..k with correct provenance; used_chunks == query output."""
    _patch_query(monkeypatch, list(CHUNKS))
    fake = FakeProvider()

    result = answer("q", provider=fake)

    citations = result["citations"]
    assert len(citations) == len(CHUNKS)
    for n, (cit, chunk) in enumerate(zip(citations, CHUNKS), 1):
        assert cit["n"] == n, f"citation not numbered sequentially: {cit!r}"
        assert cit["filename"] == chunk["filename"]
        assert cit["path"] == chunk["path"]
        assert cit["start_line"] == chunk["start_line"]
        # Citations carry exactly the provenance keys, not the full chunk.
        assert set(cit.keys()) == {"n", "filename", "path", "start_line"}

    # used_chunks is the raw query() output, unchanged and in order.
    assert result["used_chunks"] == CHUNKS


def test_answer_returns_provider_text_verbatim(monkeypatch):
    """answer()["answer"] is exactly what the provider returned."""
    _patch_query(monkeypatch, list(CHUNKS))
    fake = FakeProvider(canned="Tokio is an async runtime [1]. Futures resolve later [2].")

    result = answer("explain", provider=fake)

    assert result["answer"] == "Tokio is an async runtime [1]. Futures resolve later [2]."


def test_empty_retrieval_skips_provider(monkeypatch):
    """When query() returns [], answer() returns the no-context sentinel and never calls the provider."""
    _patch_query(monkeypatch, [])
    fake = FakeProvider()

    result = answer("nothing matches this", provider=fake)

    assert result["answer"] == _NO_CONTEXT_ANSWER
    assert result["citations"] == []
    assert result["used_chunks"] == []
    assert fake.called is False, "provider.complete must NOT be called on empty retrieval"
    assert fake.calls == []


def test_query_receives_forwarded_params(monkeypatch):
    """answer() forwards k/since/kind/hybrid through to query()."""
    recorded = _patch_query(monkeypatch, list(CHUNKS))
    fake = FakeProvider()

    answer(
        "q",
        index_dir="/some/index",
        k=3,
        provider=fake,
        since="7d",
        kind="note",
        hybrid=True,
    )

    assert recorded["question"] == "q"
    kwargs = recorded["kwargs"]
    assert kwargs["index_dir"] == "/some/index"
    assert kwargs["k"] == 3
    assert kwargs["since"] == "7d"
    assert kwargs["kind"] == "note"
    assert kwargs["hybrid"] is True


def test_default_provider_is_claude_when_none(monkeypatch):
    """With provider=None and chunks present, answer() constructs a ClaudeProvider.

    We stub ClaudeProvider in kb.answer so no real SDK/API is touched, and prove
    answer() reaches for it (rather than silently skipping synthesis).
    """
    _patch_query(monkeypatch, list(CHUNKS))

    constructed = {"count": 0}

    class StubClaude:
        def __init__(self):
            constructed["count"] += 1

        def complete(self, system, context, question):
            return "STUBBED CLAUDE ANSWER"

    monkeypatch.setattr(answer_mod, "ClaudeProvider", StubClaude)

    result = answer("q")  # provider defaults to None -> ClaudeProvider()

    assert constructed["count"] == 1, "answer() should construct ClaudeProvider once when provider=None"
    assert result["answer"] == "STUBBED CLAUDE ANSWER"
