"""Tests for kb.inbox — bare-URL detection and in-place expansion."""

from __future__ import annotations

import json

import pytest

from kb.fetch import FetchedDoc
from kb.inbox import parse_bare_url_note, process_inbox

_TS = 1783166400.0
_URL = "https://example.com/post"


# --- parse_bare_url_note: table-driven -------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        (f"{_URL}\n", _URL),                                # bare URL
        (f"\n\n  {_URL}  \n\n", _URL),                      # surrounding blanks
        (f"[An article]({_URL})", _URL),                    # markdown link only
        (f"---\ntags: [reading]\n---\n\n{_URL}\n", _URL),   # harmless frontmatter
        (f"Check this out:\n{_URL}\n", None),               # prose + URL
        (f"{_URL}\nhttps://example.com/two\n", None),       # two URLs
        ("Just a normal note about postgres.", None),       # no URL at all
        ("", None),                                         # empty file
        ("---\ntags: [x]\n---\n", None),                    # frontmatter only
        (f"---\nkb-clipped: true\n---\n\n{_URL}\n", None),  # already expanded
        (f"---\nkb-clip-failed: true\n---\n\n{_URL}\n", None),  # given up
        ("ftp://example.com/file", None),                   # non-http scheme
    ],
)
def test_parse_bare_url_note(text, expected):
    assert parse_bare_url_note(text) == expected


# --- process_inbox ----------------------------------------------------------

@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    inbox = tmp_path / "Inbox"
    inbox.mkdir()
    return inbox


def _ok_fetcher(url, timeout=15.0):
    return FetchedDoc(title="Fetched Title", text="Full body.", url=url, fetched_at=_TS)


def _fail_fetcher(url, timeout=15.0):
    return None


def test_expands_bare_url_note_in_place(home):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
    content = note.read_text(encoding="utf-8")
    assert "kb-clipped: true" in content
    assert f"url: {_URL}" in content
    assert "Full body." in content
    # Second pass is a no-op: the marker suppresses reprocessing.
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 0


def test_never_touches_notes_with_prose(home):
    note = home / "real-note.md"
    original = "My own thoughts.\n" + _URL + "\n"
    note.write_text(original, encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 0
    assert note.read_text(encoding="utf-8") == original


def test_failure_retries_then_gives_up(home, monkeypatch, tmp_path):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")

    # Failures 1 and 2: note untouched, fail count persisted.
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    assert note.read_text(encoding="utf-8") == _URL + "\n"
    state_file = tmp_path / "kbhome" / "inbox_state.json"
    assert json.loads(state_file.read_text(encoding="utf-8"))[str(note)] == 2

    # Failure 3: note marked failed, state entry cleared, no more retries.
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    content = note.read_text(encoding="utf-8")
    assert "kb-clip-failed: true" in content
    assert _URL in content  # the URL itself must survive
    assert str(note) not in json.loads(state_file.read_text(encoding="utf-8"))
    assert parse_bare_url_note(content) is None  # marker suppresses retry


def test_success_clears_fail_count(home, tmp_path):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")
    process_inbox(str(home), fetcher=_fail_fetcher)
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
    state = json.loads(
        (tmp_path / "kbhome" / "inbox_state.json").read_text(encoding="utf-8")
    )
    assert str(note) not in state


def test_missing_folder_returns_zero(home):
    assert process_inbox(str(home / "nope"), fetcher=_ok_fetcher) == 0


def test_processes_nested_subfolders(home):
    sub = home / "2026" / "07"
    sub.mkdir(parents=True)
    (sub / "share.md").write_text(_URL + "\n", encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
