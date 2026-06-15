"""Acceptance tests for kb semantic retrieval.

Two levels of guarantee:

1. ``test_rust_async_ranks_first`` — file-level: across the fixture notes, a
   Chinese query for the Rust async runtime surfaces the dedicated Rust note
   at (or near) the top.
2. ``test_chunk_excerpt_returns_rust_passage`` — chunk-level: within a single
   long note covering several unrelated topics, the same query returns the
   *Rust paragraph* as the excerpt, not some other section of the same file.

NOTE: The first run downloads the ~80MB MiniLM sentence-transformers model,
so these tests may be slow the first time. Subsequent runs use the cached
model and are fast.
"""

import os
import shutil
import time
from pathlib import Path

from kb.ingest import ingest
from kb.query import query

# Newline helper for the Stage 1 acceptance tests below.
NL = chr(10)

# Build the fixtures path from this file's location so the tests work
# regardless of pytest's current working directory.
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

EXPECTED_KEYS = {"filename", "path", "excerpt", "score", "start_line"}

# Single-topic fixture notes that are clearly unrelated to Rust async.
UNRELATED_FILES = {
    "sourdough.md",
    "tax-deadlines.md",
    "hiking-gear.txt",
    "postgres-indexing.md",
}


def test_rust_async_ranks_first(tmp_path):
    """Ingest the fixtures, query in Chinese, and rank the Rust note at the top."""
    index_dir = str(tmp_path / "idx")

    count = ingest(str(FIXTURES_DIR), index_dir=index_dir)
    # Chunking means the long mixed-topics note alone yields several chunks, so
    # there are strictly more chunks than the six fixture files.
    assert count > 6, f"Expected more chunks than files, got {count}"

    results = query("Rust 异步运行时", index_dir=index_dir, k=10)

    assert 0 < len(results) <= 10, f"Expected 1..10 results, got {len(results)}"
    for r in results:
        assert EXPECTED_KEYS.issubset(r.keys()), (
            f"Result missing keys: {EXPECTED_KEYS - set(r.keys())}"
        )

    ranking = [(r["filename"], round(float(r["score"]), 4)) for r in results]

    # The dedicated Rust note ranks at the very top (only the Rust section of
    # the mixed-topics note can plausibly compete with it).
    top_two = {r["filename"] for r in results[:2]}
    assert "rust-async.md" in top_two, (
        f"Expected rust-async.md in the top 2, got ranking: {ranking}"
    )

    # And it clearly beats every unrelated single-topic note that appears.
    best_per_file: dict[str, float] = {}
    for r in results:
        best_per_file.setdefault(r["filename"], float(r["score"]))
    rust_score = best_per_file.get("rust-async.md")
    assert rust_score is not None, f"rust-async.md absent from ranking: {ranking}"
    for name in UNRELATED_FILES & best_per_file.keys():
        assert rust_score > best_per_file[name], (
            f"rust-async.md ({rust_score:.4f}) did not beat {name} "
            f"({best_per_file[name]:.4f})\nFull ranking: {ranking}"
        )


def test_chunk_excerpt_returns_rust_passage(tmp_path):
    """Within one multi-topic note, the Rust query returns the Rust chunk."""
    # Ingest a directory holding only the long mixed-topics note, so the only
    # competition is between chunks of that single file.
    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copy(FIXTURES_DIR / "mixed-topics.md", docs / "mixed-topics.md")

    index_dir = str(tmp_path / "idx")
    count = ingest(str(docs), index_dir=index_dir)
    assert count > 1, f"Expected the long note to split into chunks, got {count}"

    results = query("Rust 异步运行时", index_dir=index_dir, k=3)
    assert results, "Expected at least one result"

    top = results[0]
    assert EXPECTED_KEYS.issubset(top.keys())
    assert top["filename"] == "mixed-topics.md"

    excerpt = top["excerpt"].lower()
    # The winning excerpt is the Rust passage...
    assert any(marker in excerpt for marker in ("tokio", "async", ".await")), (
        f"Top excerpt is not the Rust passage: {top['excerpt']!r}"
    )
    # ...and not the far-away sourdough section of the same file.
    assert "sourdough" not in excerpt, (
        f"Top excerpt leaked an unrelated section: {top['excerpt']!r}"
    )


def test_since_window_filters_old_notes(tmp_path):
    """A --since window excludes an old note even when it is the stronger match.

    Both notes are squarely about Rust async, and the OLDER one is the richer,
    denser passage — so by raw cosine it would tend to rank at least as high as
    the recent note. The only thing that can keep it out of a windowed result is
    the recency filter, which is exactly what this test pins down.
    """
    docs = tmp_path / "docs"
    docs.mkdir()

    recent_path = docs / "rust-async-recent.md"
    old_path = docs / "rust-async-old.md"

    # Recent note: short, on-topic Rust/tokio/async paragraph.
    recent_path.write_text(
        """# Rust async (recent)

A quick note on Rust async: the tokio runtime drives async tasks and you .await futures to run them concurrently.
""",
        encoding="utf-8",
    )

    # Old note: longer, denser, *more* on-topic Rust/tokio/async/.await passage,
    # so semantics alone would favour it over the recent note.
    old_path.write_text(
        """# Rust async runtime (old, in depth)

Rust async programming centers on the tokio runtime, an async executor that schedules and polls futures. You write async fn, compose futures, and .await them so the tokio runtime can drive many concurrent async tasks on its worker threads. Async Rust leans on the Future trait, the tokio scheduler, async I/O, and .await points that yield back to the tokio async runtime. tokio tokio async async .await future runtime concurrency Rust async runtime.
""",
        encoding="utf-8",
    )

    # Set mtimes explicitly against a single fixed reference so the window is
    # deterministic: recent note is "today", old note is 60 days ago.
    now = time.time()
    os.utime(recent_path, (now, now))
    os.utime(old_path, (now - 60 * 86400, now - 60 * 86400))

    index_dir = str(tmp_path / "idx")
    count = ingest(str(docs), index_dir=index_dir)
    assert count >= 2, f"Expected both notes to be indexed, got {count} chunks"

    # WITH window: only chunks modified within the last 7 days survive.
    results_recent = query("Rust 异步", index_dir=index_dir, since="7d", k=10)
    assert results_recent, "Expected at least one in-window result"
    recent_names = {r["filename"] for r in results_recent}

    assert "rust-async-recent.md" in recent_names, (
        f"Recent note missing from windowed results: {sorted(recent_names)}"
    )
    # The old note is excluded purely by recency, even though its denser passage
    # would otherwise be competitive on cosine similarity.
    assert "rust-async-old.md" not in recent_names, (
        f"Old note leaked past the --since=7d window: {sorted(recent_names)}"
    )

    # Each result carries a human-readable date string.
    for r in results_recent:
        assert isinstance(r["date"], str) and r["date"], (
            f"Result lacks a date string: {r!r}"
        )

    # WITHOUT window (baseline): both notes are retrievable, proving the old one
    # was removed by the filter rather than dropped during ingestion.
    results_all = query("Rust 异步", index_dir=index_dir, k=10)
    all_names = {r["filename"] for r in results_all}
    assert "rust-async-recent.md" in all_names, (
        f"Recent note missing from unfiltered results: {sorted(all_names)}"
    )
    assert "rust-async-old.md" in all_names, (
        f"Old note missing from unfiltered results (ingestion issue?): "
        f"{sorted(all_names)}"
    )


def test_ingest_handles_real_folder(tmp_path):
    """Ingest a realistic folder: index code/notes, skip ignored dirs and binaries."""
    proj = tmp_path / "proj"
    proj.mkdir()

    # A Python file with a unique, distinctive function name (indexed as code).
    py_file = proj / "calc_util.py"
    py_file.write_text(
        """def supercalifragilistic_widget_total(items):
    \"\"\"Sum widget totals for the supercalifragilistic report.\"\"\"
    return sum(items)
""",
        encoding="utf-8",
    )

    # A plain note with distinctive prose (indexed as a note).
    md_file = proj / "readme.md"
    md_file.write_text(
        """# Project notes

This document tracks the marmalade inventory ledger for Q3, recording every jar
counted, reconciled, and shelved across the warehouse.
""",
        encoding="utf-8",
    )

    # A JS file inside node_modules — the ignored dir must be pruned entirely.
    nm_file = proj / "node_modules" / "junk.js"
    nm_file.parent.mkdir(parents=True)
    nm_file.write_text(
        "const marmaladeNodeModulesShouldNotIndex = true;\n",
        encoding="utf-8",
    )

    # A pseudo-binary file with a .py suffix but a NUL byte — the binary guard
    # (not the suffix filter) must exclude it.
    blob_file = proj / "blob.py"
    blob_file.write_bytes(b"\x00\x01\x02 binary supercalifragilistic \x00 not text")

    index_dir = str(tmp_path / "idx")

    # Ingest must not raise despite the ignored dir and binary file.
    count = ingest(str(proj), index_dir=index_dir)
    assert count >= 1, f"Expected at least 1 chunk indexed, got {count}"

    # The unique function name surfaces calc_util.py as kind='code'.
    py_results = query("supercalifragilistic_widget_total", index_dir=index_dir, k=10)
    py_filenames = [r["filename"] for r in py_results]
    assert "calc_util.py" in py_filenames, (
        f"calc_util.py not found in results: {py_filenames}"
    )
    py_hit = next(r for r in py_results if r["filename"] == "calc_util.py")
    assert py_hit["kind"] == "code", (
        f"Expected kind='code' for calc_util.py, got {py_hit['kind']!r}"
    )

    # The note prose surfaces readme.md as kind='note'.
    md_results = query("marmalade inventory ledger", index_dir=index_dir, k=10)
    md_filenames = [r["filename"] for r in md_results]
    assert "readme.md" in md_filenames, (
        f"readme.md not found in results: {md_filenames}"
    )
    md_hit = next(r for r in md_results if r["filename"] == "readme.md")
    assert md_hit["kind"] == "note", (
        f"Expected kind='note' for readme.md, got {md_hit['kind']!r}"
    )

    # Nothing from the pruned node_modules dir, and no binary file, may appear.
    results_all = query("supercalifragilistic marmalade", index_dir=index_dir, k=50)
    names = {r["filename"] for r in results_all}
    assert "junk.js" not in names, (
        f"node_modules/junk.js was indexed but should have been pruned: {names}"
    )
    assert "blob.py" not in names, (
        f"Binary blob.py was indexed but the NUL-byte guard should skip it: {names}"
    )
    for r in results_all:
        assert "node_modules" not in r["path"], (
            f"A result path leaked the ignored node_modules dir: {r['path']!r}"
        )


def test_incremental_ingest_reuses_unchanged(tmp_path, monkeypatch):
    """Incremental ingest reuses unchanged vectors and only re-embeds changes.

    Walks five phases: first ingest, no-op re-ingest (no encode at all),
    content+mtime change on one file (only that file re-embedded, the other's
    vectors reused value-for-value), deletion (chunks vanish), and rebuild=True
    (everything present re-embedded).
    """
    import numpy
    from kb import embedding
    from kb import store

    calls = []  # list of lists-of-texts, one entry per encode invocation
    real_encode = embedding.encode

    def recording_encode(texts):
        calls.append(list(texts))
        return real_encode(texts)

    monkeypatch.setattr(embedding, "encode", recording_encode)

    docs = tmp_path / "docs"
    docs.mkdir()
    file_a = docs / "alpha.md"
    file_b = docs / "beta.md"
    file_a.write_text(
        """Alpha note about distributed consensus and raft leadership.
""",
        encoding="utf-8",
    )
    file_b.write_text(
        """Beta note about gardening compost and tomato seedlings.
""",
        encoding="utf-8",
    )
    index_dir = str(tmp_path / "idx")

    # (a) First ingest: something must be embedded.
    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    assert calls, "first ingest should embed something"
    vecs1, metas1 = store.load(index_dir)

    # (b) Re-ingest with no changes: encode must not be called and the on-disk
    # index must be byte-for-byte identical.
    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    assert calls == [], (
        f"unchanged re-ingest must not re-embed, got {calls}"
    )
    vecs2, metas2 = store.load(index_dir)
    assert numpy.array_equal(vecs1, vecs2), "vectors changed despite no edits"
    assert metas1 == metas2, "metas changed despite no edits"

    # (c) Modify one file's content and bump its mtime; only it is re-embedded.
    beta_rows_before = [i for i, m in enumerate(metas2) if m["filename"] == "beta.md"]
    assert beta_rows_before, "beta.md rows missing before the alpha edit"
    beta_vecs_before = vecs2[beta_rows_before]

    file_a.write_text(
        """Alpha note REWRITTEN: byzantine fault tolerance and quorum voting.
""",
        encoding="utf-8",
    )
    future = time.time() + 10
    os.utime(file_a, (future, future))

    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    embedded_texts = [t for c in calls for t in c]
    assert embedded_texts, "changed file should have been embedded"
    assert all("byzantine" in t or "Alpha" in t for t in embedded_texts), (
        f"only alpha.md chunks should be re-embedded, got: {embedded_texts}"
    )

    vecs3, metas3 = store.load(index_dir)
    beta_rows_after = [i for i, m in enumerate(metas3) if m["filename"] == "beta.md"]
    assert beta_rows_after, "beta.md rows missing after the alpha edit"
    beta_vecs_after = vecs3[beta_rows_after]
    assert numpy.array_equal(beta_vecs_before, beta_vecs_after), (
        "unchanged beta.md vectors were not reused value-for-value"
    )

    # (d) Delete a file: its chunks must vanish, the other survives.
    file_b.unlink()
    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    vecs4, metas4 = store.load(index_dir)
    assert all(m["filename"] != "beta.md" for m in metas4), (
        "deleted beta.md still present in index"
    )
    assert any(m["filename"] == "alpha.md" for m in metas4), (
        "alpha.md should remain after beta.md deletion"
    )

    # (e) rebuild=True re-embeds everything present.
    file_b.write_text(
        """Beta note about gardening compost and tomato seedlings.
""",
        encoding="utf-8",
    )
    ingest(str(docs), index_dir=index_dir)  # pick up the recreated beta incrementally
    calls.clear()
    ingest(str(docs), index_dir=index_dir, rebuild=True)
    embedded = [t for c in calls for t in c]
    assert any("byzantine" in t or "Alpha" in t for t in embedded), (
        f"alpha not re-embedded on rebuild, got: {embedded}"
    )
    assert any(
        ("gardening" in t or "Beta" in t or "compost" in t) for t in embedded
    ), f"beta not re-embedded on rebuild, got: {embedded}"


def test_status_reports_index_stats(tmp_path):
    """status() summarizes an existing index and reports {"exists": False} when absent."""
    from kb.status import status

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(
        "# Wombat husbandry" + NL + NL
        + "A short note about caring for wombats in captivity, covering burrows and diet." + NL,
        encoding="utf-8",
    )
    (docs / "module.py").write_text(
        "def aardvark_termite_counter(mounds):" + NL
        + "    return len(mounds)  # aardvark termite survey helper" + NL,
        encoding="utf-8",
    )

    index_dir = str(tmp_path / "idx")
    ingest(str(docs), index_dir=index_dir)

    st = status(index_dir)
    assert st["exists"] is True, f"expected an existing index, got {st!r}"
    assert st["files"] == 2, f"expected 2 source files, got {st['files']}"
    assert st["chunks"] >= 2, f"expected >=2 chunks, got {st['chunks']}"
    assert st["kinds"].get("code", 0) >= 1, (
        f"expected at least one 'code' chunk, got kinds={st['kinds']!r}"
    )
    assert st["kinds"].get("note", 0) >= 1, (
        f"expected at least one 'note' chunk, got kinds={st['kinds']!r}"
    )
    assert st["index_bytes"] > 0, f"expected non-empty index on disk, got {st['index_bytes']}"
    assert st["last_ingest"] is not None, "expected a last_ingest timestamp"

    st2 = status(str(tmp_path / "nope"))
    assert st2["exists"] is False, f"nonexistent index should report exists=False, got {st2!r}"


def test_query_reports_matched_terms(tmp_path):
    """query() result dicts expose the query terms that appear in the excerpt."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "runtime.md").write_text(
        "# Concurrency note" + NL + NL
        + "The tokio async runtime drives futures." + NL,
        encoding="utf-8",
    )

    index_dir = str(tmp_path / "idx")
    ingest(str(docs), index_dir=index_dir)

    results = query("tokio async runtime", index_dir=index_dir, k=5)
    assert results, "expected at least one result for the tokio query"

    top = results[0]
    assert "matched_terms" in top, f"result missing 'matched_terms': {sorted(top.keys())}"
    assert set(top["matched_terms"]) & {"tokio", "async", "runtime"}, (
        f"expected overlap, got {top['matched_terms']}"
    )
    expected = {"filename", "path", "excerpt", "score", "start_line", "kind", "matched_terms"}
    assert expected <= set(top.keys()), (
        f"result missing keys: {expected - set(top.keys())}"
    )


def test_ingest_decodes_utf16_skips_binary(tmp_path):
    """UTF-16 text is decoded and indexed; NUL-byte binaries (even with text suffixes) are skipped."""
    docs = tmp_path / "docs"
    docs.mkdir()

    # UTF-16 file: BOM + embedded NUL bytes, yet it must STILL be indexed.
    (docs / "u16.md").write_bytes(("UTF16 note about marmoset primates." + NL).encode("utf-16"))

    # Real binary with a non-text suffix (filtered by suffix anyway).
    (docs / "blob.dat").write_bytes(bytes([0, 1]) + b"rawbinary" + bytes([0]) + b"stuff")

    # Binary with a SUPPORTED suffix but NUL bytes and no BOM: the NUL guard
    # (not the suffix filter) must skip it.
    (docs / "fake.md").write_bytes(bytes([0, 1]) + b" not really text " + bytes([0]))

    index_dir = str(tmp_path / "idx")
    count = ingest(str(docs), index_dir=index_dir)
    assert count >= 1, f"expected at least one chunk indexed, got {count}"

    results = query("marmoset primates", index_dir=index_dir, k=5)
    assert any(r["filename"] == "u16.md" for r in results), (
        f"UTF-16 file should be indexed and findable, got {[r['filename'] for r in results]}"
    )

    fake_results = query("not really text", index_dir=index_dir, k=10)
    assert all(r["filename"] != "fake.md" for r in fake_results), (
        f"NUL-byte fake.md should have been skipped, got {[r['filename'] for r in fake_results]}"
    )
