"""Acceptance tests for the BookmarkSource + ``ingest-bookmarks`` command.

These mirror the conventions in ``test_query.py``:

* ``FIXTURES_DIR`` / ``REPO_ROOT`` are derived from ``__file__`` so the tests
  run regardless of pytest's current working directory.
* Ingest goes through the same public seam the CLI handler uses --
  ``_ingest_from_source(BookmarkSource(path), index_dir=...)`` -- and search
  through ``kb.query.query``.
* Incremental reuse is asserted by monkeypatching ``kb.embedding.encode`` to
  record every batch of texts it is asked to embed (the recording-encode
  pattern from ``test_query.py``).

NOTE: the first run downloads the MiniLM sentence-transformers model, so these
tests may be slow the first time and fast thereafter (cached model).
"""

import json
import subprocess
import sys
from pathlib import Path

from kb.ingest import _ingest_from_source
from kb.query import query
from kb.source import BookmarkSource

# Built from this file's location so cwd never matters (see test_query.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
BOOKMARKS_FIXTURE = FIXTURES_DIR / "bookmarks" / "Bookmarks"

# Standard result-dict shape every query() hit must carry.
EXPECTED_KEYS = {"filename", "path", "excerpt", "score", "start_line"}

# Distinctive identifiers baked into the fixture.
RUST_TITLE = "Rust async book - Tokio tutorial"
RUST_URL = "https://tokio.rs/tokio/tutorial"


def test_bookmark_ingest_and_query(tmp_path):
    """Ingest the Chrome fixture and surface the Rust/Tokio bookmark."""
    index_dir = str(tmp_path / "idx")

    count = _ingest_from_source(BookmarkSource(BOOKMARKS_FIXTURE), index_dir=index_dir)
    assert count > 0, f"expected at least one chunk indexed, got {count}"

    results = query("Tokio async runtime tutorial", index_dir=index_dir, k=10)
    assert results, "expected at least one result for the Tokio query"

    for r in results:
        assert EXPECTED_KEYS.issubset(r.keys()), (
            f"result missing keys: {EXPECTED_KEYS - set(r.keys())}"
        )

    # The Rust/Tokio bookmark surfaces: match on filename==title OR path==url.
    assert any(
        r["filename"] == RUST_TITLE or r["path"] == RUST_URL for r in results
    ), (
        "Rust/Tokio bookmark not found in results: "
        f"{[(r['filename'], r['path']) for r in results]}"
    )


def test_bookmark_incremental_reuse(tmp_path, monkeypatch):
    """A no-op re-ingest of a non-file source must not call encode at all."""
    from kb import embedding

    calls = []  # one entry (a list of texts) per encode invocation
    real_encode = embedding.encode

    def recording_encode(texts):
        calls.append(list(texts))
        return real_encode(texts)

    monkeypatch.setattr(embedding, "encode", recording_encode)

    index_dir = str(tmp_path / "idx")

    # First ingest: something must be embedded.
    calls.clear()
    _ingest_from_source(BookmarkSource(BOOKMARKS_FIXTURE), index_dir=index_dir)
    assert calls, "first ingest should embed something"

    # Second, unchanged ingest: change_token reuse means NO encode at all.
    calls.clear()
    _ingest_from_source(BookmarkSource(BOOKMARKS_FIXTURE), index_dir=index_dir)
    assert calls == [], (
        f"unchanged bookmark re-ingest must not re-embed, got {calls}"
    )


def _write_two_bookmarks(path, first_title):
    """Write a minimal Chrome Bookmarks file with two url nodes."""
    data = {
        "checksum": "0",
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "name": "Bookmarks bar",
                "date_added": "13350000000000000",
                "children": [
                    {
                        "type": "url",
                        "name": first_title,
                        "url": "https://example.com/changing",
                        "guid": "guid-changing-001",
                        "date_added": "13350000000000001",
                    },
                    {
                        "type": "url",
                        "name": "Stable kombucha brewing notes",
                        "url": "https://example.com/kombucha",
                        "guid": "guid-stable-002",
                        "date_added": "13350000000000002",
                    },
                ],
            }
        },
        "version": 1,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bookmark_changed_reembeds_only_changed(tmp_path, monkeypatch):
    """Editing one bookmark's title re-embeds only that bookmark."""
    from kb import embedding

    calls = []
    real_encode = embedding.encode

    def recording_encode(texts):
        calls.append(list(texts))
        return real_encode(texts)

    monkeypatch.setattr(embedding, "encode", recording_encode)

    bm_path = tmp_path / "Bookmarks"
    index_dir = str(tmp_path / "idx")

    original_title = "Original quantum computing primer"
    new_title = "Revised neuromorphic hardware primer"
    stable_title = "Stable kombucha brewing notes"

    # First ingest of both bookmarks.
    _write_two_bookmarks(bm_path, original_title)
    _ingest_from_source(BookmarkSource(bm_path), index_dir=index_dir)

    # Change ONLY the first bookmark's title; keep the second identical
    # (same guid/url/date_added), so its change_token is unchanged.
    _write_two_bookmarks(bm_path, new_title)

    calls.clear()
    _ingest_from_source(BookmarkSource(bm_path), index_dir=index_dir)

    embedded_texts = [t for c in calls for t in c]
    assert embedded_texts, "the changed bookmark should have been re-embedded"
    # The new title is embedded; the unchanged bookmark's title is NOT.
    assert any(new_title in t for t in embedded_texts), (
        f"changed bookmark's new title was not re-embedded: {embedded_texts}"
    )
    assert all(stable_title not in t for t in embedded_texts), (
        f"unchanged bookmark was re-embedded but should have been reused: "
        f"{embedded_texts}"
    )
    # And the original title is gone for good.
    assert all(original_title not in t for t in embedded_texts), (
        f"stale original title leaked into the re-embed: {embedded_texts}"
    )


def test_cli_ingest_bookmarks_smoke(tmp_path):
    """Drive the real CLI: ingest-bookmarks then query, via subprocess."""
    index_dir = str(tmp_path / "idx")

    ingest_proc = subprocess.run(
        [
            sys.executable, "-m", "kb", "ingest-bookmarks",
            str(BOOKMARKS_FIXTURE), "--index-dir", index_dir,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert ingest_proc.returncode == 0, (
        f"ingest-bookmarks failed (rc={ingest_proc.returncode})\n"
        f"STDOUT: {ingest_proc.stdout}\nSTDERR: {ingest_proc.stderr}"
    )
    assert "Indexed" in ingest_proc.stdout, (
        f"expected 'Indexed' in ingest stdout, got: {ingest_proc.stdout!r}"
    )

    query_proc = subprocess.run(
        [
            sys.executable, "-m", "kb", "query",
            "sourdough bread", "--index-dir", index_dir,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert query_proc.returncode == 0, (
        f"query failed (rc={query_proc.returncode})\n"
        f"STDOUT: {query_proc.stdout}\nSTDERR: {query_proc.stderr}"
    )
    out = query_proc.stdout.lower()
    assert "sourdough" in out, (
        f"sourdough bookmark not found in query stdout: {query_proc.stdout!r}"
    )
