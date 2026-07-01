"""Lexical (keyword) scoring for the kb knowledge base.

A small, dependency-free BM25 implementation used to complement the dense
vector search with exact keyword matching. Pure standard library -- no numpy,
no external ranking libraries -- so it stays as light as the rest of the store.

BM25 rewards documents that contain rare query terms frequently while damping
the contribution of very long documents. Scores are non-negative; a document
sharing no terms with the query scores ``0.0``.
"""

from __future__ import annotations

import math
import re

# CJK Unicode ranges treated as single-character tokens. Mirrors ingest's
# chunk tokenizer (kb/ingest.py) so the BM25 side segments Chinese/Japanese/
# Korean the same way the dense side does -- otherwise a space-free CJK run
# collapses into one giant token that never recurs and matches nothing.
_CJK_CHARS = (
    r"一-鿿"   # CJK Unified Ideographs
    r"㐀-䶿"   # CJK Extension A
    r"豈-﫿"   # CJK Compatibility Ideographs
    r"぀-ゟ"   # Hiragana
    r"゠-ヿ"   # Katakana
    r"가-힯"   # Hangul Syllables
)

# A "token" is either a single CJK character or a run of word characters that
# are NOT CJK (``[^\W...]`` is "a word char, but not one of these"). Lower-cased
# for case-insensitive matching; punctuation is dropped.
_TOKEN_RE = re.compile(rf"[{_CJK_CHARS}]|[^\W{_CJK_CHARS}]+", re.UNICODE)


_CJK_CHAR_RE = re.compile(rf"[{_CJK_CHARS}]")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-cased tokens (CJK at character granularity)."""
    return _TOKEN_RE.findall(text.lower())


def _is_cjk_char(tok: str) -> bool:
    return len(tok) == 1 and _CJK_CHAR_RE.match(tok) is not None


def bm25_terms(text: str) -> list[str]:
    """Tokenize for BM25, shingling consecutive CJK characters into bigrams.

    Single Chinese/Japanese/Korean characters are far too common to be useful
    keywords (``的``/``是``/``用`` appear everywhere), so a run of CJK characters
    is emitted as overlapping 2-character shingles -- ``语音转文字`` becomes
    ``语音``, ``音转``, ``转文``, ``文字``. Latin/digit tokens pass through
    unchanged; an isolated single CJK character (no neighbour to pair with) is
    kept as-is so it can still match.
    """
    terms: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            terms.append(run[0])
        else:
            terms.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        run.clear()

    for tok in tokenize(text):
        if _is_cjk_char(tok):
            run.append(tok)
        else:
            flush()
            terms.append(tok)
    flush()
    return terms


def bm25_scores(
    question: str,
    docs: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return a BM25 score per doc in ``docs`` for ``question``.

    ``docs`` are raw chunk texts. Scores are ``>= 0``; a doc with no
    query-term overlap scores ``0.0``. An empty query yields all zeros, and
    empty (or all-empty) docs never crash.

    Args:
        question: The natural-language query.
        docs: Candidate documents (raw chunk texts), scored in place.
        k1: Term-frequency saturation parameter (higher -> less saturation).
        b: Length-normalization parameter (0 disables, 1 fully normalizes).

    Returns:
        A list of floats aligned to ``docs`` order.
    """
    # Unique, non-empty query terms. No terms or no docs -> nothing to score.
    q_terms = {t for t in bm25_terms(question) if t}
    if not q_terms or not docs:
        return [0.0] * len(docs)

    n = len(docs)

    # Tokenize each doc once; keep parallel lengths for normalization.
    doc_tokens = [bm25_terms(d) for d in docs]
    doc_len = [len(toks) for toks in doc_tokens]

    avgdl = sum(doc_len) / n
    # All docs empty -> avgdl is 0; length normalization would divide by zero.
    if avgdl == 0:
        return [0.0] * len(docs)

    # Document frequency of each query term across the candidate docs.
    df: dict[str, int] = {term: 0 for term in q_terms}
    for toks in doc_tokens:
        for term in q_terms.intersection(toks):
            df[term] += 1

    # BM25+ idf form: stays non-negative even for very common terms.
    idf = {
        term: math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for term in q_terms
    }

    scores: list[float] = []
    for i, toks in enumerate(doc_tokens):
        # Term frequencies for just the query terms present in this doc.
        tf: dict[str, int] = {}
        for tok in toks:
            if tok in q_terms:
                tf[tok] = tf.get(tok, 0) + 1

        denom_norm = k1 * (1 - b + b * doc_len[i] / avgdl)
        score = 0.0
        for term, f in tf.items():
            score += idf[term] * (f * (k1 + 1)) / (f + denom_norm)
        scores.append(score)

    return scores
