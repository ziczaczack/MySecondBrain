"""Acceptance tests for the kb.serve local web UI.

The server is started for real on an ephemeral localhost port and driven with
``urllib``, so routing, JSON shapes, status codes, and the token gate are
exercised end to end over a socket -- no mocked transport.

Retrieval and synthesis are stubbed by monkeypatching ``kb.serve.query`` /
``kb.serve.answer`` (same convention as ``test_answer.py``), so no MiniLM model
loads and no API call is made.
"""

from __future__ import annotations

import contextlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from kb import serve as serve_mod


def _chunk(filename="rust-async.md", start_line=12, **extra):
    """A result dict shaped like kb.query.query() output."""
    base = {
        "filename": filename,
        "path": f"/notes/{filename}",
        "text": "The tokio runtime drives async tasks to completion.",
        "excerpt": "The tokio runtime drives async tasks to completion.",
        "start_line": start_line,
        "score": 0.5,
        "mtime": 1.0,
        "date": "2026-08-02",
        "kind": "note",
        "matched_terms": ["tokio"],
    }
    base.update(extra)
    return base


@contextlib.contextmanager
def running(token=None):
    """Start the server on an ephemeral port; yield its base URL; shut it down."""
    server = serve_mod.make_server(
        index_dir="/does/not/matter", host="127.0.0.1", port=0, token=token
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(url, payload=None):
    """Return (status, body_text) for a GET, or a JSON POST when payload is given."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def test_search_endpoint_returns_results_as_json(monkeypatch):
    """GET /api/search runs retrieval and returns the result dicts as JSON."""
    recorded = {}

    def fake_query(question, **kwargs):
        recorded["question"] = question
        recorded["kwargs"] = kwargs
        return [_chunk()]

    monkeypatch.setattr(serve_mod, "query", fake_query)

    with running() as base:
        status, body = _request(base + "/api/search?q=tokio+runtime&k=3&hybrid=1")

    assert status == 200, body
    payload = json.loads(body)
    assert [r["filename"] for r in payload["results"]] == ["rust-async.md"]
    assert payload["results"][0]["start_line"] == 12
    # Query options survive the trip through the query string.
    assert recorded["question"] == "tokio runtime"
    assert recorded["kwargs"]["k"] == 3
    assert recorded["kwargs"]["hybrid"] is True


def test_ask_endpoint_returns_answer_and_citations(monkeypatch):
    """POST /api/ask synthesizes and returns the answer plus its citations."""

    def fake_answer(question, **kwargs):
        return {
            "answer": "Tokio drives async tasks [1].",
            "citations": [
                {"n": 1, "filename": "rust-async.md",
                 "path": "/notes/rust-async.md", "start_line": 12}
            ],
            "used_chunks": [_chunk()],
        }

    monkeypatch.setattr(serve_mod, "answer", fake_answer)

    with running() as base:
        status, body = _request(base + "/api/ask", {"question": "what is tokio?"})

    assert status == 200, body
    payload = json.loads(body)
    assert payload["answer"] == "Tokio drives async tasks [1]."
    assert payload["citations"][0]["filename"] == "rust-async.md"


def test_ask_endpoint_reports_provider_errors_as_json(monkeypatch):
    """A KbLLMError (e.g. no API key) becomes a readable JSON error, not a 500 traceback."""

    def boom(question, **kwargs):
        raise serve_mod.KbLLMError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(serve_mod, "answer", boom)

    with running() as base:
        status, body = _request(base + "/api/ask", {"question": "q"})

    assert status == 400, body
    assert "ANTHROPIC_API_KEY" in json.loads(body)["error"]


def test_index_page_is_served_and_self_contained():
    """GET / returns an HTML page that pulls in no external resources."""
    with running() as base:
        status, body = _request(base + "/")

    assert status == 200
    assert "<!doctype html>" in body.lower()
    # A strict local-first page: nothing may be fetched from the network.
    for offender in ("http://", "https://", "//cdn", "src=\"//"):
        assert offender not in body.replace("obsidian://", ""), (
            f"page references an external resource: {offender!r}"
        )


def test_unknown_path_returns_404():
    with running() as base:
        status, _ = _request(base + "/nope")
    assert status == 404


def test_token_gate_rejects_requests_without_the_token(monkeypatch):
    """When a token is configured every request must carry it."""
    monkeypatch.setattr(serve_mod, "query", lambda question, **kw: [_chunk()])

    with running(token="s3cret") as base:
        denied, _ = _request(base + "/api/search?q=tokio")
        allowed, body = _request(base + "/api/search?q=tokio&t=s3cret")
        wrong, _ = _request(base + "/api/search?q=tokio&t=nope")

    assert denied == 401
    assert wrong == 401
    assert allowed == 200, body


def test_clip_endpoint_saves_and_indexes(monkeypatch, tmp_path):
    """POST /api/clip fetches the URL, writes a note, and reindexes the clips folder."""
    from kb.fetch import FetchedDoc

    clips = tmp_path / "Clips"
    indexed = []

    monkeypatch.setattr(serve_mod.config, "clips_dir", lambda: str(clips))
    monkeypatch.setattr(
        serve_mod,
        "fetch_url",
        lambda url, timeout=15.0: FetchedDoc(
            title="Partial Indexes", text="body text", url=url, fetched_at=1.0
        ),
    )
    monkeypatch.setattr(serve_mod, "ingest", lambda d, index_dir=None: indexed.append(d))

    with running() as base:
        status, body = _request(
            base + "/api/clip", {"url": "https://example.com/post"}
        )

    assert status == 200, body
    saved = json.loads(body)["path"]
    assert saved.endswith(".md")
    assert "Partial Indexes" in (clips / "Partial-Indexes.md").read_text(encoding="utf-8")
    assert indexed == [str(clips)], "a new clip must be indexed immediately"


def test_status_endpoint_reports_sources_and_counts(monkeypatch):
    """GET /api/status answers "what am I actually searching?" without a terminal."""
    monkeypatch.setattr(
        serve_mod,
        "index_status",
        lambda index_dir: {
            "exists": True,
            "index_dir": index_dir,
            "files": 38,
            "chunks": 87,
            "kinds": {"note": 78, "code": 9},
            "index_bytes": 274227,
            "last_ingest_date": "2026-07-13 00:37",
        },
    )
    monkeypatch.setattr(
        serve_mod.config,
        "load_sources",
        lambda: [{"kind": "files", "path": r"D:\KnowledgeBase"}],
    )
    monkeypatch.setattr(serve_mod.config, "clips_dir", lambda: r"D:\KnowledgeBase\Clips")

    with running() as base:
        status, body = _request(base + "/api/status")

    assert status == 200, body
    payload = json.loads(body)
    assert payload["files"] == 38 and payload["chunks"] == 87
    assert payload["sources"] == [{"kind": "files", "path": r"D:\KnowledgeBase"}]
    assert payload["clips_dir"] == r"D:\KnowledgeBase\Clips"
    assert payload["last_ingest_date"] == "2026-07-13 00:37"


def test_reindex_endpoint_reingests_every_registered_folder(monkeypatch):
    """POST /api/reindex re-ingests each registered folder and reports what ran.

    It deliberately does not report a chunk total: ingest() returns the size of
    the whole index after its pass, not the number of chunks it contributed, so
    summing across folders that share an index would multiply-count. The client
    re-reads /api/status for counts instead.
    """
    monkeypatch.setattr(serve_mod, "_files_folders", lambda: ["/notes", "/code"])
    seen = []
    monkeypatch.setattr(
        serve_mod, "ingest", lambda folder, index_dir=None: seen.append(folder)
    )

    with running() as base:
        status, body = _request(base + "/api/reindex", {})

    assert status == 200, body
    payload = json.loads(body)
    assert seen == ["/notes", "/code"]
    assert payload["folders"] == 2
    assert payload["failed"] == []
    assert "chunks" not in payload


def test_reindex_reports_a_failing_folder_without_aborting(monkeypatch):
    """One unreadable source must not take down the whole reindex."""
    monkeypatch.setattr(serve_mod, "_files_folders", lambda: ["/gone", "/ok"])
    seen = []

    def fake_ingest(folder, index_dir=None):
        if folder == "/gone":
            raise OSError("vanished")
        seen.append(folder)

    monkeypatch.setattr(serve_mod, "ingest", fake_ingest)

    with running() as base:
        status, body = _request(base + "/api/reindex", {})

    assert status == 200, body
    payload = json.loads(body)
    assert seen == ["/ok"], "the healthy folder must still be indexed"
    assert payload["folders"] == 1, "only the folder that succeeded is counted"
    assert [f["path"] for f in payload["failed"]] == ["/gone"]
    assert "vanished" in payload["failed"][0]["error"]


def test_reindex_with_no_sources_says_so(monkeypatch):
    monkeypatch.setattr(serve_mod, "_files_folders", lambda: [])

    with running() as base:
        status, body = _request(base + "/api/reindex", {})

    assert status == 400, body
    assert "kb add" in json.loads(body)["error"]
