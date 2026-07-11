"""Tests for capture-layer config: clips_dir resolution and the inbox kind."""

from __future__ import annotations

import json

from kb import config


def test_clips_dir_unconfigured_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    assert config.clips_dir() is None


def test_clips_dir_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.setenv("KB_CLIPS_DIR", "D:/vault/Clips")
    assert config.clips_dir() == "D:/vault/Clips"


def test_set_clips_dir_persists_and_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    config.set_clips_dir(str(tmp_path / "Clips"))
    assert config.clips_dir() == str(tmp_path / "Clips")


def test_set_clips_dir_preserves_other_config_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_MODEL", raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model": "claude-sonnet-5"}), encoding="utf-8")
    config.set_clips_dir(str(tmp_path / "Clips"))
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["model"] == "claude-sonnet-5"
    assert data["clips_dir"] == str(tmp_path / "Clips")
    # synthesis_model still resolves from the same file.
    assert config.synthesis_model() == "claude-sonnet-5"


def test_set_clips_dir_recovers_from_corrupt_config(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text("not json{", encoding="utf-8")
    config.set_clips_dir(str(tmp_path / "Clips"))
    assert config.clips_dir() == str(tmp_path / "Clips")


def test_add_inbox_source(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    sources = config.add_source("inbox", str(tmp_path / "Inbox"))
    assert sources == [{"kind": "inbox", "path": str(tmp_path / "Inbox")}]
