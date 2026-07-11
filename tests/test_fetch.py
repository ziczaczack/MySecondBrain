"""Tests for kb.fetch — no real network access anywhere in this file."""

from __future__ import annotations

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
    result = fetch.extract_article(ARTICLE_HTML)
    assert result is not None
    title, text = result
    assert "Partial Indexes" in title
    assert "unshipped orders" in text
    # Boilerplate must not leak into the extracted text.
    assert "Copyright" not in text


def test_extract_article_rejects_empty_html():
    assert fetch.extract_article("<html><body></body></html>") is None


def test_fetch_url_rejects_non_http_schemes():
    assert fetch.fetch_url("file:///etc/passwd") is None
    assert fetch.fetch_url("notaurl") is None


def test_fetch_url_article_path(monkeypatch):
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
