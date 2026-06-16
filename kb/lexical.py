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

# A "token" is any run of word characters, lower-cased for case-insensitive
# matching. Unicode-aware so non-ASCII notes tokenize sensibly.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-cased word tokens."""
    return _TOKEN_RE.findall(text.lower())


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
    q_terms = {t for t in tokenize(question) if t}
    if not q_terms or not docs:
        return [0.0] * len(docs)

    n = len(docs)

    # Tokenize each doc once; keep parallel lengths for normalization.
    doc_tokens = [tokenize(d) for d in docs]
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
