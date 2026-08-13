"""Tests for kb.fetch — no real network access anywhere in this file."""

from __future__ import annotations

import pytest

import kb.fetch as fetch

# Realistic-enough article HTML: trafilatura needs a few substantial
# paragraphs before it recognises a main-content block.
ARTICLE_HTML = """<!DOCTYPE html>
<html><head><title>Postgres Partial Indexes Explained</title></head>
<body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<article>
<h1>Postgres Partial Indexes Explained</h1>
<p>A partial index covers only the rows that satisfy a predicate, which makes
it dramatically smaller than a full index when the predicate is selective.
This paragraph exists to give the extractor a substantial block of prose to
recognise as main content rather than boilerplate navigation.</p>
<p>The classic example is indexing only unshipped orders: the index stays tiny
because shipped orders fall out of it automatically. Queries that repeat the
predicate in their WHERE clause can use the index; queries that do not repeat
it cannot, which is the most common surprise for newcomers.</p>
<p>Partial indexes also combine well with UNIQUE, letting you enforce
uniqueness over a subset of rows, such as at most one active subscription per
customer. That trick is hard to express any other way and is worth remembering
when modelling soft-deleted records.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>"""


# Same shape as ARTICLE_HTML but with the structure that plain-text extraction
# destroys: a section heading and an ordered list whose numbering carries meaning.
STRUCTURED_HTML = """<!DOCTYPE html>
<html><head><title>Deploying With Zero Downtime</title></head>
<body>
<nav><a href="/">Home</a></nav>
<article>
<h1>Deploying With Zero Downtime</h1>
<p>Rolling deploys keep a service available while its code changes underneath,
by replacing instances a few at a time instead of all at once. This paragraph
exists to give the extractor a substantial block of prose to recognise as main
content rather than boilerplate navigation.</p>
<p>The tricky part is not the rollout itself but the database, because old and
new code run simultaneously against one schema. Every migration therefore has
to be compatible with the version of the code that is about to be replaced as
well as the one replacing it.</p>
<h2>Migration Order</h2>
<p>The steps have to happen in this order:</p>
<ol>
<li><strong>Add the new column</strong> as nullable, so existing writes keep working.</li>
<li><strong>Backfill</strong> it in batches, so the table is never locked for long.</li>
<li><strong>Start writing</strong> both columns from the new code path.</li>
<li><strong>Drop the old column</strong> only once no running code reads it.</li>
</ol>
</article>
<footer>Copyright 2026</footer>
</body></html>"""


def test_youtube_video_id_matches_watch_and_short_urls():
    assert fetch.youtube_video_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"
    assert fetch.youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert fetch.youtube_video_id(
        "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"


def test_youtube_video_id_rejects_non_youtube():
    assert fetch.youtube_video_id("https://example.com/watch?v=abcdef1") is None
    assert fetch.youtube_video_id("https://vimeo.com/12345") is None


def test_extract_article_returns_title_and_text():
    # extract_article() returns None both when trafilatura is absent and when a
    # page has no main content, so without the library this test cannot tell a
    # real regression from a missing optional dep. Skip rather than fail red.
    pytest.importorskip("trafilatura")
    result = fetch.extract_article(ARTICLE_HTML)
    assert result is not None
    title, text = result
    assert "Partial Indexes" in title
    assert "unshipped orders" in text
    # Boilerplate must not leak into the extracted text.
    assert "Copyright" not in text


def test_extract_article_keeps_headings_as_markdown():
    # Section headings are what make a long clip navigable in Obsidian: the
    # outline pane, folding, and [[note#heading]] links all key off them.
    pytest.importorskip("trafilatura")
    result = fetch.extract_article(STRUCTURED_HTML)
    assert result is not None
    _, text = result
    heading = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    assert any("Migration Order" in ln for ln in heading), (
        f"no markdown heading for the <h2>; got headings {heading!r}"
    )


def test_extract_article_keeps_ordered_lists_numbered():
    # An <ol> flattened to bullets loses the sequence, and in a migration or a
    # how-to the sequence *is* the content.
    pytest.importorskip("trafilatura")
    result = fetch.extract_article(STRUCTURED_HTML)
    assert result is not None
    _, text = result
    assert "1." in text and "4." in text, (
        "ordered list lost its numbering (flattened to bullets?)"
    )


def test_extract_article_rejects_empty_html():
    assert fetch.extract_article("<html><body></body></html>") is None


def test_fetch_url_rejects_non_http_schemes():
    assert fetch.fetch_url("file:///etc/passwd") is None
    assert fetch.fetch_url("notaurl") is None


def test_fetch_url_article_path(monkeypatch):
    pytest.importorskip("trafilatura")
    monkeypatch.setattr(
        fetch, "_http_get", lambda url, timeout: ARTICLE_HTML.encode("utf-8")
    )
    doc = fetch.fetch_url("https://example.com/post")
    assert doc is not None
    assert doc.url == "https://example.com/post"
    assert "unshipped orders" in doc.text
    assert doc.fetched_at > 0


def test_fetch_url_returns_none_on_network_failure(monkeypatch):
    monkeypatch.setattr(fetch, "_http_get", lambda url, timeout: None)
    assert fetch.fetch_url("https://example.com/post") is None


def test_fetch_url_youtube_path(monkeypatch):
    monkeypatch.setattr(fetch, "_fetch_transcript", lambda vid: "hello transcript")
    monkeypatch.setattr(fetch, "_youtube_title", lambda url, timeout: "Video Title")
    doc = fetch.fetch_url("https://youtu.be/dQw4w9WgXcQ")
    assert doc is not None
    assert doc.title == "Video Title"
    assert doc.text == "hello transcript"


def test_fetch_url_youtube_no_transcript(monkeypatch):
    monkeypatch.setattr(fetch, "_fetch_transcript", lambda vid: None)
    assert fetch.fetch_url("https://youtu.be/dQw4w9WgXcQ") is None
