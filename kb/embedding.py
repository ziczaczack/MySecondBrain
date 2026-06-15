from __future__ import annotations

import numpy as np

_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformers model once and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings, one row per input text.

    Shape: (len(texts), 384).
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def encode_one(text: str) -> np.ndarray:
    """Return a single 1D float32 embedding for one text."""
    return encode([text])[0]
