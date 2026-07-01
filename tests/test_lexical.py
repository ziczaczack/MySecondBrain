"""Tests for CJK-aware lexical tokenization and BM25 keyword scoring.

The dense embedding side already chunks CJK at character granularity; the BM25
side must tokenize the same way or Chinese queries match nothing (a whole
space-free Chinese run collapses into one giant token that never recurs).
"""

from kb.lexical import bm25_scores, tokenize


def test_tokenize_splits_cjk_into_single_characters():
    assert tokenize("语音转文字") == ["语", "音", "转", "文", "字"]


def test_tokenize_keeps_latin_runs_and_lowercases():
    assert tokenize("Tokio async_runtime") == ["tokio", "async_runtime"]


def test_tokenize_mixed_cjk_and_latin_separates_at_boundary():
    # A Latin run must not swallow the following CJK characters.
    assert tokenize("Polymarket交易") == ["polymarket", "交", "易"]


def test_tokenize_drops_pure_punctuation():
    assert tokenize("a, b.") == ["a", "b"]


def test_bm25_uses_cjk_bigrams_not_just_single_characters():
    """Single common characters must not match; only real 2-char phrases do.

    "量子" is distinctive as a bigram. A doc containing 量 and 子 only inside
    unrelated words (数量, 孔子) -- never adjacent as 量子 -- must score zero,
    or every long Chinese note full of common characters becomes a false match.
    """
    scores = bm25_scores("量子", ["量子纠缠理论", "数量众多与孔子思想"])
    assert scores[0] > 0, f"the doc with the real 量子 bigram must match, got {scores}"
    assert scores[1] == 0.0, (
        f"a doc with 量 and 子 only in unrelated words must not match, got {scores}"
    )


def test_bm25_rewards_cjk_character_overlap():
    """A doc sharing the query's Chinese characters must outscore an unrelated one."""
    q = "语音转文字"
    docs = [
        "这是一个语音转文字的桌面应用程序",  # shares every query char
        "完全无关的烹饪食谱与园艺内容",        # shares (almost) none
    ]
    scores = bm25_scores(q, docs)
    assert scores[0] > scores[1], (
        f"CJK-overlapping doc must score higher, got {scores}"
    )
    assert scores[0] > 0, f"expected positive BM25 for the matching doc, got {scores}"
