"""Acceptance tests for the zero-config workflow.

These pin down the managed-home behaviour that lets ``kb`` run with no explicit
``--index-dir``: a per-user home (redirectable via ``$KB_HOME``), a default
index location under that home, a JSON source registry, a polling watcher, and
the ``add`` / ``ask`` / ``sources`` CLI subcommands.

Conventions mirror ``test_query.py`` / ``test_bookmarks.py``: REPO_ROOT and
FIXTURES_DIR are derived from ``__file__``; in-process tests redirect the
managed home via ``KB_HOME`` before any config/watch/CLI call; subprocess tests
pass an os.environ copy with KB_HOME and PYTHONIOENCODING=utf-8.

NOTE: the first run downloads the MiniLM model, so the roundtrip / watch /
subprocess tests may be slow the first time and fast thereafter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"


def _child_env(kb_home: Path) -> dict:
    """Return an os.environ copy with KB_HOME redirected and UTF-8 forced."""
    env = os.environ.copy()
    env["KB_HOME"] = str(kb_home)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_default_index_dir_follows_kb_home(tmp_path, monkeypatch):
    """default_index_dir() lives under the redirected home; registry starts empty."""
    home = tmp_path / "kbhome"
    monkeypatch.setenv("KB_HOME", str(home))

    from kb import config

    index_dir = Path(config.default_index_dir())
    assert index_dir.name == "index", f"expected an 'index' leaf, got {index_dir!r}"
    assert home in index_dir.parents, (
        f"default index {index_dir!r} is not under the redirected home {home!r}"
    )
    assert config.load_sources() == [], (
        f"expected an empty registry for a fresh home, got {config.load_sources()!r}"
    )


def test_add_source_registers_and_dedups(tmp_path, monkeypatch):
    """add_source registers a files source, dedups repeats, and rejects bad kinds."""
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))

    from kb import config

    notes = tmp_path / "notes"
    notes.mkdir()

    config.add_source("files", str(notes))
    srcs = config.load_sources()
    files_entries = [s for s in srcs if s.get("kind") == "files"]
    assert len(files_entries) == 1, f"expected exactly one files entry, got {srcs!r}"
    assert files_entries[0]["path"] == str(Path(notes)), (
        f"registered path not normalized as expected: {files_entries[0]!r}"
    )

    config.add_source("files", str(notes))
    srcs2 = config.load_sources()
    files_entries2 = [s for s in srcs2 if s.get("kind") == "files"]
    assert len(files_entries2) == 1, (
        f"re-adding the same source duplicated it: {srcs2!r}"
    )

    with pytest.raises(ValueError):
        config.add_source("bogus", str(notes))


def test_add_then_ask_roundtrip(tmp_path, monkeypatch):
    """add a folder, ingest into the managed default index, then ask finds the note.

    Proves the add -> ask path works against the managed default index with no
    explicit --index-dir: the same default_index_dir() is used for both writing
    and reading.
    """
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))

    from kb import config
    from kb.ingest import ingest
    from kb.query import query

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "marmoset.md").write_text(
        "# Marmoset care\n\n"
        "These are detailed marmoset primate husbandry notes covering diet, "
        "enclosure temperature, and social grouping.\n",
        encoding="utf-8",
    )

    config.add_source("files", str(notes))
    count = ingest(str(notes), index_dir=config.default_index_dir())
    assert count >= 1, f"expected at least one chunk indexed, got {count}"

    results = query("marmoset primate", index_dir=config.default_index_dir(), k=5)
    assert results, "expected at least one result for the marmoset query"
    assert any(r["filename"] == "marmoset.md" for r in results), (
        f"marmoset.md did not surface: {[r['filename'] for r in results]}"
    )


def test_watch_run_once_cycle(tmp_path, monkeypatch):
    """run_once reports change on first run + new file, and no-change when idle."""
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))

    from kb import config, watch

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "first.md").write_text(
        "# First note\n\nA note about distributed consensus and raft leadership.\n",
        encoding="utf-8",
    )

    config.add_source("files", str(notes))

    changed, st = watch.run_once()
    assert changed is True, (
        "first run over a registered, non-empty folder must report change"
    )

    changed2, st2 = watch.run_once(state=st)
    assert changed2 is False, "an idle re-run must report no change"

    (notes / "second.md").write_text(
        "# Second note\n\nA note about gardening compost and tomato seedlings.\n",
        encoding="utf-8",
    )
    changed3, st3 = watch.run_once(state=st2)
    assert changed3 is True, (
        "adding a new .md file must report change on the next cycle"
    )


def test_cli_add_ask_sources_subprocess(tmp_path):
    """End-to-end CLI smoke: sources -> add -> sources -> ask, via subprocess.

    A fresh KB_HOME in the child env keeps every call off the real managed home.
    """
    env = _child_env(tmp_path / "kbhome")
    base = [sys.executable, "-m", "kb"]

    def run(args):
        return subprocess.run(
            base + args,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    p_empty = run(["sources"])
    assert p_empty.returncode == 0, (
        f"sources failed (rc={p_empty.returncode})\n"
        f"STDOUT: {p_empty.stdout}\nSTDERR: {p_empty.stderr}"
    )
    assert "No sources registered" in p_empty.stdout, (
        f"expected 'No sources registered' message, got: {p_empty.stdout!r}"
    )

    p_add = run(["add", str(FIXTURES_DIR)])
    assert p_add.returncode == 0, (
        f"add failed (rc={p_add.returncode})\n"
        f"STDOUT: {p_add.stdout}\nSTDERR: {p_add.stderr}"
    )
    assert "Added files source" in p_add.stdout, (
        f"expected 'Added files source' confirmation, got: {p_add.stdout!r}"
    )

    p_list = run(["sources"])
    assert p_list.returncode == 0, (
        f"sources (post-add) failed (rc={p_list.returncode})\n"
        f"STDOUT: {p_list.stdout}\nSTDERR: {p_list.stderr}"
    )
    assert "[files]" in p_list.stdout, (
        f"expected a '[files]' entry after add, got: {p_list.stdout!r}"
    )

    # Use --no-synthesis so the CLI smoke test has no network dependency.
    # Synthesis (LLM integration) is tested separately with a mock provider.
    p_ask = run(["ask", "--no-synthesis", "Rust 异步运行时"])
    assert p_ask.returncode == 0, (
        f"ask failed (rc={p_ask.returncode})\n"
        f"STDOUT: {p_ask.stdout}\nSTDERR: {p_ask.stderr}"
    )
    assert "rust" in p_ask.stdout.lower(), (
        f"expected a Rust hit in ask output, got: {p_ask.stdout!r}"
    )

    p_ask_json = run(["ask", "--no-synthesis", "--json", "Rust 异步运行时"])
    assert p_ask_json.returncode == 0, (
        f"ask --json failed (rc={p_ask_json.returncode})\n"
        f"STDOUT: {p_ask_json.stdout}\nSTDERR: {p_ask_json.stderr}"
    )
    import json as _json
    payload = _json.loads(p_ask_json.stdout)
    assert isinstance(payload, list), f"expected a JSON list, got: {type(payload)}"
