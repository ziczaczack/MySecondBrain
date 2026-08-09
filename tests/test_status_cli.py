"""CLI-level tests for `kb status`.

The key assertion is that status reports whether an API key is configured:
otherwise the only way to discover a missing key is to run `kb ask` and have it
fail, which is a worse way to learn it.
"""

from __future__ import annotations

import sys

import numpy
import pytest

import kb.__main__ as cli
from kb import store


def _run(argv, monkeypatch):
    """Invoke the real CLI entrypoint; return its SystemExit code."""
    monkeypatch.setattr(sys, "argv", ["kb"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    return exc_info.value.code


@pytest.fixture
def index_dir(tmp_path):
    """A minimal real index, so status() reports exists=True without embedding."""
    target = tmp_path / "idx"
    store.save(
        numpy.ones((1, 4), dtype=numpy.float32),
        [{"path": "/n/a.md", "filename": "a.md", "chunk_text": "x", "kind": "note"}],
        str(target),
    )
    return str(target)


def test_status_reports_a_missing_api_key(monkeypatch, tmp_path, index_dir, capsys):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _run(["status", "--index-dir", index_dir], monkeypatch) == 0

    out = capsys.readouterr().out
    assert "not configured" in out, f"status hid the missing key:\n{out}"


def test_status_reports_a_configured_api_key(monkeypatch, tmp_path, index_dir, capsys):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    assert _run(["status", "--index-dir", index_dir], monkeypatch) == 0

    out = capsys.readouterr().out
    assert "env:ANTHROPIC_API_KEY" in out, f"status did not name the key source:\n{out}"
    assert "sk-ant-not-a-real-key" not in out, "the key value must never be printed"
