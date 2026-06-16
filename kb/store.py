"""Numpy-based vector store for the kb knowledge base.

A minimal, dependency-light vector store backed by plain numpy arrays and
JSON metadata. No FAISS, no sqlite-vec -- just numpy and the standard library.

An "index" lives in a directory containing two files:
  - vectors.npy : a 2D float array of shape (n_vectors, dim)
  - meta.json   : a JSON list of metadata dicts, one per vector
"""

import json
import os

import numpy

# Filenames used within an index directory.
_VECTORS_FILE = "vectors.npy"
_META_FILE = "meta.json"

# Small constant to guard against division by zero when normalizing.
_EPSILON = 1e-12


def save(vectors: numpy.ndarray, metas: list[dict], index_dir: str) -> None:
    """Persist vectors and their metadata to ``index_dir``.

    Args:
        vectors: 2D array of shape (n_vectors, dim).
        metas: List of metadata dicts, one per vector.
        index_dir: Directory to write the index into (created if missing).

    Raises:
        ValueError: If ``len(vectors)`` does not equal ``len(metas)``.
    """
    if len(vectors) != len(metas):
        raise ValueError(
            f"Number of vectors ({len(vectors)}) does not match "
            f"number of metadata entries ({len(metas)})."
        )

    os.makedirs(index_dir, exist_ok=True)

    vectors_path = os.path.join(index_dir, _VECTORS_FILE)
    meta_path = os.path.join(index_dir, _META_FILE)

    numpy.save(vectors_path, vectors)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)


def load(index_dir: str) -> tuple[numpy.ndarray, list[dict]]:
    """Load vectors and metadata from ``index_dir``.

    Args:
        index_dir: Directory containing the index files.

    Returns:
        A tuple of (vectors, metas).

    Raises:
        FileNotFoundError: If either index file is missing.
    """
    vectors_path = os.path.join(index_dir, _VECTORS_FILE)
    meta_path = os.path.join(index_dir, _META_FILE)

    if not os.path.exists(vectors_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No index found in '{index_dir}'. "
            f"Expected '{_VECTORS_FILE}' and '{_META_FILE}'. "
            f"Run `python -m kb ingest <dir>` first to build the index."
        )

    vectors = numpy.load(vectors_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    return vectors, metas


def cosine_scores(
    query_vec: numpy.ndarray, vectors: numpy.ndarray
) -> numpy.ndarray:
    """Cosine similarity of ``query_vec`` against every row of ``vectors``.

    Args:
        query_vec: 1D query vector.
        vectors: 2D matrix of stored vectors, shape (n_vectors, dim).

    Returns:
        A 1D float array of length ``len(vectors)``; an empty ``vectors``
        yields an empty array.
    """
    n_vectors = len(vectors)
    if n_vectors == 0:
        return numpy.zeros(0)

    # Work in float to avoid integer truncation during normalization.
    query = numpy.asarray(query_vec, dtype=numpy.float64).ravel()
    matrix = numpy.asarray(vectors, dtype=numpy.float64)

    # Defensive normalization (rows may or may not already be unit-length).
    query_norm = numpy.linalg.norm(query)
    query_unit = query / max(query_norm, _EPSILON)

    matrix_norms = numpy.linalg.norm(matrix, axis=1)
    matrix_norms = numpy.maximum(matrix_norms, _EPSILON)
    matrix_unit = matrix / matrix_norms[:, numpy.newaxis]

    # Cosine similarity reduces to a dot product on normalized vectors.
    return matrix_unit @ query_unit


def search(
    query_vec: numpy.ndarray, vectors: numpy.ndarray, k: int = 5
) -> list[tuple[int, float]]:
    """Return the top-``k`` most similar vectors by cosine similarity.

    Args:
        query_vec: 1D query vector.
        vectors: 2D matrix of stored vectors, shape (n_vectors, dim).
        k: Number of results to return. If ``k`` exceeds the number of
            stored vectors, all of them are returned.

    Returns:
        A list of (index, score) tuples sorted by descending similarity.
    """
    n_vectors = len(vectors)
    if n_vectors == 0 or k <= 0:
        return []

    # Cosine similarity against every stored vector.
    scores = cosine_scores(query_vec, vectors)

    k = min(k, n_vectors)

    # Partial selection of the top-k, then sort just those by score.
    top_unsorted = numpy.argpartition(-scores, k - 1)[:k]
    top_sorted = top_unsorted[numpy.argsort(-scores[top_unsorted])]

    return [(int(i), float(scores[i])) for i in top_sorted]
