"""Tests for kb.ingest._chunk_document — the window that feeds the embedder.

Chunk boundaries decide what a citation can point at, so they are worth
testing directly rather than only through the query layer.
"""

from __future__ import annotations

import re

# Imported by name: `kb.ingest` resolves to the ingest() *function*, which
# kb/__init__.py re-exports over the submodule of the same name.
from kb.ingest import _chunk_document

# Seven tokens each, so a 200-word window lands mid-sentence (200 = 28×7 + 4)
# rather than falling on a boundary by luck.
SENTENCES = [
    f"Sentence number {i} contains exactly seven words." for i in range(1, 61)
]
LONG_PROSE = " ".join(SENTENCES)

FRONTMATTER_NOTE = (
    "---\n"
    'title: "Zero Downtime Deploys"\n'
    "url: https://example.com/deploys\n"
    "clipped: 2026-08-11\n"
    "kb-clipped: true\n"
    "---\n"
    "\n"
    "Rolling deploys replace instances a few at a time.\n"
)


def test_chunking_omits_yaml_frontmatter():
    # Frontmatter is identity, not prose. Embedding it wastes window on every
    # clip and makes "kb-clipped" a term that matches every clipped note.
    chunks = _chunk_document(FRONTMATTER_NOTE)
    assert chunks, "expected at least one chunk"
    body = chunks[0][0]
    assert "Rolling deploys" in body
    for leaked in ("kb-clipped", "clipped:", "url:", "title:"):
        assert leaked not in body, f"frontmatter key {leaked!r} leaked into {body!r}"


def test_chunking_reports_line_numbers_from_the_original_file():
    # Skipping frontmatter must not shift start_line, or citations point at the
    # wrong line. The body of FRONTMATTER_NOTE begins on line 8.
    chunks = _chunk_document(FRONTMATTER_NOTE)
    assert chunks[0][1] == 8


def test_chunking_a_frontmatter_only_note_yields_nothing():
    only_meta = "---\ntitle: \"Empty\"\nkb-clipped: true\n---\n"
    assert _chunk_document(only_meta) == []


def test_leading_horizontal_rule_is_not_treated_as_frontmatter():
    # A document opening with a rule and no closing delimiter is prose, and
    # stripping to the next "---" would silently eat real content.
    content = "---\n\nA thematic break opened this note. " + LONG_PROSE
    chunks = _chunk_document(content)
    assert "A thematic break opened this note." in chunks[0][0]


def test_chunks_end_on_sentence_boundaries():
    # A chunk cut mid-sentence splits a term from its definition, which is how
    # a bold list label ends up in a different chunk than the text defining it.
    chunks = _chunk_document(LONG_PROSE)
    assert len(chunks) > 1, "fixture too short to exercise windowing"
    for text, _ in chunks[:-1]:
        assert text.rstrip().endswith("."), f"chunk cut mid-sentence: ...{text[-60:]!r}"


def test_chunks_do_not_end_on_a_dangling_list_marker():
    # "2." looks like a sentence ending but is an ordered-list marker. Snapping
    # there strands the number in one chunk and its item text in the next —
    # the same label/definition split that boundary snapping exists to prevent.
    steps = "\n".join(
        f"{i}. **Step {i} label:** and the prose that explains what it does."
        for i in range(1, 41)
    )
    chunks = _chunk_document(steps)
    assert len(chunks) > 1, "fixture too short to exercise windowing"
    for text, _ in chunks[:-1]:
        last = text.rstrip().rsplit(None, 1)[-1]
        assert not re.fullmatch(r"\d+[.)]", last), (
            f"chunk ends on a dangling list marker {last!r}"
        )


def test_chunking_still_covers_the_whole_document():
    # Boundary snapping must not drop content between windows.
    chunks = _chunk_document(LONG_PROSE)
    joined = " ".join(text for text, _ in chunks)
    for sentence in (SENTENCES[0], SENTENCES[29], SENTENCES[-1]):
        assert sentence in joined, f"lost {sentence!r}"
