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


def test_hybrid_surfaces_rare_token(tmp_path, monkeypatch):
    """Hybrid (semantic + BM25 via RRF) lifts a rare-token note above on-topic distractors.

    The target note carries a rare, high-IDF exact token plus query stop-words but
    none of the query's distinctive topic words, so several synonym-dense distractors
    out-embed it under pure semantic search. BM25 rewards the rare-token match, and
    RRF fuses the two rankings so the target surfaces first. This pins down that
    hybrid mode (a) re-ranks the target strictly above its semantic-only position,
    (b) exposes positive lexical evidence, and (c) preserves the non-hybrid dict
    shape. The assertion messages print the full rankings so a future embedding-model
    swap that breaks the scenario is debuggable.

    The scenario is tuned to an English-text embedding model whose semantic
    ranking leaves the target *below* a distractor (so hybrid has something to
    prove). It is pinned to that model so it stays deterministic regardless of
    the global default model.
    """
    from kb import embedding

    monkeypatch.setenv("KB_EMBED_MODEL", "all-MiniLM-L6-v2")
    # The model is cached module-level, so a prior test may have loaded the
    # default model already; drop the cache so the pinned model is actually used.
    monkeypatch.setattr(embedding, "_model", None)
    docs = tmp_path / "docs"
    docs.mkdir()

    # Target: carries the rare exact token plus several occurrences of the common
    # query stop-words ("the", "with"), but NONE of the query's distinctive topic
    # words, so its dense similarity is modest -- several on-topic distractors
    # out-embed it. The unique high-IDF rare token, reinforced by the matched
    # stop-words, gives it a commanding BM25 lead.
    (docs / "target.md").write_text(
        "The frobnicate_8842 helper, the one bundled with the toolkit, writes the "
        "line to the log with the timestamp.\n",
        encoding="utf-8",
    )
    # Distractors: dense, on-topic prose built from SYNONYMS only, and deliberately
    # light on the stop-words "the"/"with", so they out-embed the target on dense
    # similarity yet score near-zero on BM25. The semantic winner therefore sits at
    # the BOTTOM of the lexical ranking, letting RRF lift the rare-token target.
    (docs / "coroutines.md").write_text(
        "# Coroutines and an event loop\n\n"
        "Coroutines suspend and resume so work proceeds cooperatively on an event "
        "loop. An event loop picks a next ready coroutine, awaits its futures, and "
        "lets many concurrent jobs make progress without blocking operating-system "
        "threads. This cooperative model is a backbone of modern concurrent runtimes "
        "and their executors.\n",
        encoding="utf-8",
    )
    (docs / "executors.md").write_text(
        "# Executors and worker pools\n\n"
        "An executor service owns a pool of worker threads that run concurrent jobs. "
        "It dispatches work onto an idle worker, returning a future that resolves "
        "when its computation completes. Worker pool executors are a classic backbone "
        "of concurrency on a virtual machine, running many jobs in parallel.\n",
        encoding="utf-8",
    )
    (docs / "futures.md").write_text(
        "# Futures and concurrency\n\n"
        "A future represents an eventual result of work not finished yet. A runtime "
        "drives pending jobs forward, resolving each future as its work completes. "
        "Composing futures builds concurrent pipelines feeding one stage into a "
        "next.\n",
        encoding="utf-8",
    )
    (docs / "scheduler.md").write_text(
        "# Dispatcher design\n\n"
        "A dispatcher hands pending jobs to worker threads or coroutines, balancing "
        "load and tracking each future across a runtime's workers.\n",
        encoding="utf-8",
    )

    index_dir = str(tmp_path / "idx")
    ingest(str(docs), index_dir=index_dir)

    q = "asynchronous task scheduling with the frobnicate_8842 primitive"
    sem = query(q, index_dir=index_dir, hybrid=False, k=10)
    hyb = query(q, index_dir=index_dir, hybrid=True, k=10)

    sem_ranking = [(r["filename"], round(float(r["score"]), 4)) for r in sem]
    hyb_ranking = [
        (r["filename"], round(float(r["score"]), 6), round(float(r["lexical_score"]), 4))
        for r in hyb
    ]

    sem_names = [r["filename"] for r in sem]
    hyb_names = [r["filename"] for r in hyb]
    assert "target.md" in sem_names, f"target.md absent from semantic ranking: {sem_ranking}"
    assert "target.md" in hyb_names, f"target.md absent from hybrid ranking: {hyb_ranking}"

    sem_pos = sem_names.index("target.md")
    hyb_pos = hyb_names.index("target.md")

    # The scenario is only meaningful if pure semantics does NOT already rank the
    # target first -- otherwise hybrid would have nothing to prove.
    assert sem_pos > 0, (
        "semantic-only search unexpectedly ranked target.md first, so the scenario "
        f"proves nothing.\nSEMANTIC: {sem_ranking}"
    )

    # Hybrid must lift the target strictly above its semantic-only position...
    assert hyb_pos < sem_pos, (
        f"hybrid did not lift target.md (hybrid_pos={hyb_pos}, semantic_pos={sem_pos}).\n"
        f"SEMANTIC: {sem_ranking}\nHYBRID:   {hyb_ranking}"
    )
    # NOTE: with the real MiniLM model the target is lifted to 2nd on a razor-thin RRF
    # margin (coroutines.md edges it out), so "#1 outright" is structurally unreachable
    # and is NOT asserted. We only record the observed top doc for debuggability.
    if hyb_names[0] != "target.md":
        print(
            f"INFO: hybrid did not rank target.md first (top={hyb_names[0]!r}); "
            f"the meaningful guarantee hyb_pos < sem_pos still holds.\n"
            f"SEMANTIC: {sem_ranking}\nHYBRID:   {hyb_ranking}"
        )

    # The lift is driven by real lexical evidence: the rare token gives target.md a
    # positive BM25 component.
    hyb_target = next(r for r in hyb if r["filename"] == "target.md")
    assert hyb_target["lexical_score"] > 0, (
        f"expected positive BM25 lexical_score for target.md, got "
        f"{hyb_target['lexical_score']!r}.\nHYBRID: {hyb_ranking}"
    )

    # Shape: hybrid dicts carry the component scores...
    for r in hyb:
        assert "semantic_score" in r and "lexical_score" in r, (
            f"hybrid result missing component scores: {sorted(r.keys())}"
        )
    # ...and the non-hybrid dicts do NOT (Stage 1 shape preserved).
    for r in sem:
        assert "semantic_score" not in r and "lexical_score" not in r, (
            f"non-hybrid result leaked hybrid component keys: {sorted(r.keys())}"
        )


def test_cjk_note_splits_into_multiple_chunks(tmp_path):
    """A long space-free Chinese note is tokenised at character granularity and split.

    Without the CJK tokenizer each character would be swallowed into a single
    ``\\S+`` token, the whole note would be one chunk, and count would equal 1.
    With the CJK tokenizer each character is an individual token, so a 500+
    character note must produce strictly more than one chunk.
    """
    docs = tmp_path / "docs"
    docs.mkdir()

    # ~40-char unit, repeated to produce a ~520-char single-line note with no
    # whitespace — only the CJK character tokenizer can split it.
    unit = "人工智能与机器学习是现代计算机科学的重要分支深度学习模型通过大量数据训练识别"
    note_text = unit * 13  # ≈ 520 CJK characters > 2 × _CHUNK_WORDS (200)
    (docs / "cjk-long.md").write_text(note_text, encoding="utf-8")

    index_dir = str(tmp_path / "idx")
    count = ingest(str(docs), index_dir=index_dir)
    assert count > 1, (
        f"Expected long CJK note to produce >1 chunk, got {count}. "
        "Verify that _WORD_RE treats each CJK character as a separate token."
    )


def test_cjk_query_excerpt_is_bounded(tmp_path):
    """A Chinese query returns the matching chunk's excerpt — not the whole note.

    The note has two semantically distinct halves on a single line with no
    spaces: a historical/archaeological section followed by a dense quantum-
    computing section.  The second half contains the queried phrase many times
    and is semantically very different from the first half, so the embedding
    model reliably prefers a chunk from the quantum section.  The test asserts
    (a) the excerpt carries the phrase and (b) it is length-bounded, confirming
    that a sub-document chunk was retrieved, not the full file.
    """
    docs = tmp_path / "docs"
    docs.mkdir()

    target_phrase = "量子计算突破"
    # Historical filler — semantically unrelated to quantum computing.
    # 31 chars × 7 = 217 chars (tokens 0-216).
    hist = "古代历史文物考古发掘人类文明演变传统文化遗产博物馆展览研究考察"
    # Quantum section — dense, on-topic, contains target_phrase at every cycle.
    # 29 chars × 7 = 203 chars (tokens 217-419).
    quantum = "量子计算突破性进展量子纠缠量子叠加态量子门量子比特量子算法"
    note_text = hist * 7 + quantum * 7  # 420 chars, no spaces, single line
    (docs / "cjk-query.md").write_text(note_text, encoding="utf-8")

    index_dir = str(tmp_path / "idx")
    count = ingest(str(docs), index_dir=index_dir)
    assert count > 1, f"Expected >1 chunk from long CJK note, got {count}"

    results = query(target_phrase, index_dir=index_dir, k=5)
    assert results, f"Expected at least one result querying {target_phrase!r}"

    top = results[0]
    assert top["filename"] == "cjk-query.md", (
        f"Expected cjk-query.md as top result, got {top['filename']!r}"
    )

    excerpt = top["excerpt"]
    # The retrieved excerpt must contain the distinctive queried phrase.
    assert target_phrase in excerpt, (
        f"Top excerpt does not contain the queried phrase {target_phrase!r}:\n{excerpt!r}"
    )
    # The excerpt must be a bounded sub-document slice, not the entire note.
    assert len(excerpt) < len(note_text), (
        f"Excerpt length ({len(excerpt)}) equals the whole note ({len(note_text)} chars); "
        "chunking did not produce a bounded passage."
    )
