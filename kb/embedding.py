from __future__ import annotations

import os

import numpy as np

_model = None

# Default to a multilingual model so Chinese/Japanese/Korean notes embed in the
# same space as English ones; the old English-only model (and any other) can be
# selected via $KB_EMBED_MODEL. Switching models requires a re-ingest, which the
# index's stamped model name enforces (see current_model_name / LEGACY_MODEL).
_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _model_name() -> str:
    name = os.environ.get("KB_EMBED_MODEL", "").strip()
    return name or _DEFAULT_MODEL


def current_model_name() -> str:
    """The embedding model that ingest/query will use right now.

    Stamped into the index at ingest time and checked at query time so an index
    built with one model is never searched with another.
    """
    return _model_name()


# Indexes built before the model was stamped used this model exclusively.
LEGACY_MODEL = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformers model once and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_model_name(), device="cpu")
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
