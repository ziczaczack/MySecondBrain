"""Tests for the Reciprocal Rank Fusion helper used by hybrid search.

A document with no lexical (BM25) overlap must contribute nothing from the
lexical channel, so an all-zero or sparse lexical signal can never reorder the
semantic ranking. This is what makes hybrid safe to enable by default on a
CJK-heavy corpus, where most docs share no exact tokens with the query.
"""

from kb.query import _rrf_fuse


def _order(fused):
    return sorted(range(len(fused)), key=lambda i: -fused[i])


def test_zero_lexical_channel_preserves_semantic_order():
    """When no doc has lexical overlap, fused order equals semantic order."""
    sem = [0.9, 0.5, 0.1]
    lex = [0.0, 0.0, 0.0]
    assert _order(_rrf_fuse(sem, lex)) == [0, 1, 2]


def test_zero_lexical_does_not_penalize_a_late_indexed_semantic_winner():
    """A clear semantic winner sitting last in corpus order must stay on top.

    This is the exact failure that made hybrid unusable for pure-Chinese
    queries: with no lexical signal, phantom index-order lexical ranks dragged
    the real semantic winner down the list.
    """
    sem = [0.1, 0.2, 0.9]  # doc 2 is the clear winner
    lex = [0.0, 0.0, 0.0]  # no lexical overlap anywhere
    assert _order(_rrf_fuse(sem, lex))[0] == 2


def test_positive_lexical_overlap_lifts_a_weaker_semantic_doc():
    """A doc with real lexical overlap is rewarded relative to a zero-overlap doc."""
    # doc 1 loses on semantics but is the only lexical match.
    sem = [0.9, 0.4]
    lex = [0.0, 5.0]
    fused = _rrf_fuse(sem, lex)
    # doc 1 must be lifted strictly above its semantic-only position (last).
    assert fused[1] > 1.0 / (60 + 1), "lexical overlap must add RRF credit"
