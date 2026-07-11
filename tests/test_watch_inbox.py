"""Watch-cycle integration: inbox expansion feeds the normal reingest path."""

from __future__ import annotations

import sys

import pytest

import kb.__main__ as cli
import kb.watch as watch
from kb import config
from kb.fetch import FetchedDoc

_TS = 1783166400.0
_URL = "https://example.com/post"


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    return tmp_path


def test_add_inbox_registers_and_ingests(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    ingested = []
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: ingested.append(d))
    monkeypatch.setattr(sys, "argv", ["kb", "add", "--inbox", str(inbox)])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0
    assert {"kind": "inbox", "path": str(inbox)} in config.load_sources()
    assert ingested == [str(inbox)]


def test_run_once_expands_inbox_then_reingests(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    (inbox / "share.md").write_text(_URL + "\n", encoding="utf-8")
    config.add_source("inbox", str(inbox))

    monkeypatch.setattr(
        "kb.fetch.fetch_url",
        lambda url, timeout=15.0: FetchedDoc(
            title="T", text="Body.", url=url, fetched_at=_TS
        ),
    )
    ingested = []
    monkeypatch.setattr(watch, "ingest_fn", lambda d, **kw: ingested.append(d))

    changed, state = watch.run_once(index_dir=str(home / "idx"), state={})
    assert changed is True
    assert ingested == [str(inbox)]  # inbox folders are watched as files too
    content = (inbox / "share.md").read_text(encoding="utf-8")
    assert "kb-clipped: true" in content


def test_run_once_inbox_failure_does_not_break_cycle(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    (inbox / "share.md").write_text(_URL + "\n", encoding="utf-8")
    config.add_source("inbox", str(inbox))

    def _boom(folder, timeout=15.0, fetcher=None):
        raise RuntimeError("inbox exploded")

    monkeypatch.setattr(watch, "process_inbox", _boom)
    monkeypatch.setattr(watch, "ingest_fn", lambda d, **kw: None)
    # Must not raise: the inbox step is isolated from the watch cycle.
    changed, state = watch.run_once(index_dir=str(home / "idx"), state={})
    assert changed is True  # the bare-URL note itself still counts as a file
