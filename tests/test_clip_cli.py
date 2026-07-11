"""CLI-level tests for `kb clip` — fetch and ingest are always monkeypatched."""

from __future__ import annotations

import io
import sys

import pytest

import kb.__main__ as cli
from kb import config
from kb.fetch import FetchedDoc

_TS = 1783166400.0


def _run(argv, monkeypatch):
    """Invoke the real CLI entrypoint; return its SystemExit code."""
    monkeypatch.setattr(sys, "argv", ["kb"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    return exc_info.value.code


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    return tmp_path


def test_clip_set_dir(home, monkeypatch):
    target = home / "Clips"
    assert _run(["clip", "--set-dir", str(target)], monkeypatch) == 0
    assert config.clips_dir() == str(target)


def test_clip_without_configured_dir_errors(home, monkeypatch):
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 1


def test_clip_url_saves_registers_and_ingests(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    doc = FetchedDoc(
        title="Post", text="Body.", url="https://example.com/post", fetched_at=_TS
    )
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: doc)
    ingested = []
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: ingested.append(d))

    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert list(clips.glob("*.md")), "clip note not written"
    assert ingested == [str(clips)]
    assert {"kind": "files", "path": str(clips)} in config.load_sources()


def test_clip_fetch_failure_exits_1(home, monkeypatch):
    config.set_clips_dir(str(home / "Clips"))
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: None)
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 1


def test_clip_duplicate_url_is_ok_but_skips(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    doc = FetchedDoc(
        title="Post", text="Body.", url="https://example.com/post", fetched_at=_TS
    )
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: doc)
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: None)
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert len(list(clips.glob("*.md"))) == 1


def test_clip_text_mode_reads_stdin(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("A conclusion from claude.ai"))
    assert _run(["clip", "--text", "Agent memory notes"], monkeypatch) == 0
    files = list(clips.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "A conclusion from claude.ai" in content
    assert 'title: "Agent memory notes"' in content


def test_clip_no_url_no_text_errors(home, monkeypatch):
    config.set_clips_dir(str(home / "Clips"))
    assert _run(["clip"], monkeypatch) == 1
