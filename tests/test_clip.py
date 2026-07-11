"""Tests for kb.clip — note rendering and on-disk saving."""

from __future__ import annotations

from pathlib import Path

from kb.clip import find_existing, render_note, save_clip, slug
from kb.fetch import FetchedDoc

# 2026-07-11 12:00:00 UTC — any fixed timestamp works; the date assertion
# below derives the expected string the same way the implementation does.
_TS = 1783166400.0


def _doc(title="Partial Indexes", url="https://example.com/post", text="Body text."):
    return FetchedDoc(title=title, text=text, url=url, fetched_at=_TS)


def test_slug_ascii():
    assert slug("Postgres Partial Indexes!") == "Postgres-Partial-Indexes"


def test_slug_keeps_cjk():
    assert slug("局部索引详解") == "局部索引详解"


def test_slug_truncates_to_60_chars():
    assert len(slug("x" * 200)) <= 60


def test_slug_empty_falls_back():
    assert slug("///???") == "clip"


def test_render_note_frontmatter():
    text = render_note(_doc(), "2026-07-11")
    lines = text.splitlines()
    assert lines[0] == "---"
    assert 'title: "Partial Indexes"' in lines
    assert "url: https://example.com/post" in lines
    assert "clipped: 2026-07-11" in lines
    assert "kb-clipped: true" in lines
    assert "Body text." in text


def test_render_note_escapes_double_quotes_in_title():
    text = render_note(_doc(title='He said "hi"'), "2026-07-11")
    assert "title: \"He said 'hi'\"" in text


def test_save_clip_writes_file(tmp_path):
    path = save_clip(_doc(), str(tmp_path))
    assert path is not None and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "kb-clipped: true" in content
    assert path.name == "Partial-Indexes.md"


def test_save_clip_dedupes_by_url(tmp_path):
    assert save_clip(_doc(), str(tmp_path)) is not None
    assert save_clip(_doc(title="Different Title"), str(tmp_path)) is None
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_save_clip_text_mode_never_dedupes(tmp_path):
    # Pasted-text clips have url="" and must not collide with each other.
    assert save_clip(_doc(title="Note A", url=""), str(tmp_path)) is not None
    assert save_clip(_doc(title="Note B", url=""), str(tmp_path)) is not None
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_save_clip_filename_collision_appends_counter(tmp_path):
    assert save_clip(_doc(url="https://example.com/a"), str(tmp_path)) is not None
    p2 = save_clip(_doc(url="https://example.com/b"), str(tmp_path))
    assert p2 is not None and p2.name == "Partial-Indexes-2.md"


def test_find_existing_matches_frontmatter_url(tmp_path):
    save_clip(_doc(), str(tmp_path))
    assert find_existing(Path(tmp_path), "https://example.com/post") is not None
    assert find_existing(Path(tmp_path), "https://example.com/other") is None


def test_save_clip_creates_missing_folder(tmp_path):
    target = tmp_path / "nested" / "Clips"
    assert save_clip(_doc(), str(target)) is not None
    assert target.is_dir()
