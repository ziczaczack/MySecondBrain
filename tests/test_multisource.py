"""Regression tests: multiple sources sharing one index must coexist."""

from __future__ import annotations

import json
from pathlib import Path

from kb.ingest import ingest


def _indexed_keys(index_dir) -> set[str]:
    metas = json.loads(
        (Path(index_dir) / "meta.json").read_text(encoding="utf-8")
    )
    return {m["key"] for m in metas}


def test_two_sibling_sources_coexist(tmp_path):
    idx = str(tmp_path / "idx")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "a1.md").write_text("alpha document about postgres indexes", encoding="utf-8")
    (b / "b1.md").write_text("beta document about rust lifetimes", encoding="utf-8")

    ingest(str(a), index_dir=idx)
    ingest(str(b), index_dir=idx)

    keys = _indexed_keys(idx)
    assert str(a / "a1.md") in keys, "sibling source A was clobbered by ingesting B"
    assert str(b / "b1.md") in keys


def test_nested_clips_folder_preserves_vault(tmp_path):
    idx = str(tmp_path / "idx")
    vault = tmp_path / "vault"
    clips = vault / "Clips"
    clips.mkdir(parents=True)
    (vault / "note.md").write_text("my own note about embeddings", encoding="utf-8")
    (clips / "clip1.md").write_text("clipped article body text", encoding="utf-8")

    ingest(str(vault), index_dir=idx)   # vault covers Clips too
    ingest(str(clips), index_dir=idx)   # re-ingesting the nested folder...

    keys = _indexed_keys(idx)
    assert str(vault / "note.md") in keys, "vault was clobbered by nested Clips ingest"
    assert str(clips / "clip1.md") in keys


def test_deletion_within_a_source_still_removed(tmp_path):
    idx = str(tmp_path / "idx")
    a = tmp_path / "a"
    a.mkdir()
    keep, gone = a / "keep.md", a / "gone.md"
    keep.write_text("this file stays around", encoding="utf-8")
    gone.write_text("this file will be deleted", encoding="utf-8")

    ingest(str(a), index_dir=idx)
    gone.unlink()
    ingest(str(a), index_dir=idx)

    keys = _indexed_keys(idx)
    assert str(keep) in keys
    assert str(gone) not in keys, "deleted file must still be dropped from the index"


def test_files_and_bookmarks_coexist(tmp_path):
    idx = str(tmp_path / "idx")
    a = tmp_path / "a"
    a.mkdir()
    (a / "a1.md").write_text("plain note next to bookmarks", encoding="utf-8")

    bookmarks = tmp_path / "Bookmarks"
    bookmarks.write_text(json.dumps({
        "roots": {
            "bookmark_bar": {
                "type": "folder", "name": "Bookmarks bar",
                "children": [{
                    "type": "url", "name": "Example", "guid": "guid-1",
                    "url": "https://example.com/", "date_added": "13300000000000000",
                }],
            }
        }
    }), encoding="utf-8")

    ingest(str(a), index_dir=idx)
    from kb.ingest import _ingest_from_source
    from kb.source import BookmarkSource
    _ingest_from_source(BookmarkSource(str(bookmarks)), index_dir=idx, label="bm")

    keys = _indexed_keys(idx)
    assert str(a / "a1.md") in keys, "file source was clobbered by bookmark ingest"
    assert "guid-1" in keys
