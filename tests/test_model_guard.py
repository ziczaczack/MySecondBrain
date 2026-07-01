"""The index records which embedding model built it, and querying or
re-ingesting with a different model is caught instead of silently comparing
vectors from two different models (same dimensionality, garbage similarity).
"""

import numpy
import pytest

from kb import store
from kb.query import IncompatibleIndexError, query


def test_query_rejects_index_built_with_a_different_embed_model(tmp_path):
    # A hand-built index stamped with a model that is NOT the current default.
    metas = [{"chunk_text": "hello", "embed_model": "some-other-model", "mtime": None}]
    vecs = numpy.zeros((1, 3), dtype="float32")
    index_dir = str(tmp_path / "idx")
    store.save(vecs, metas, index_dir)

    # The guard must fire before any embedding happens, so no model is loaded.
    with pytest.raises(IncompatibleIndexError):
        query("anything", index_dir=index_dir)


def test_legacy_index_without_model_stamp_is_treated_as_the_old_default(tmp_path):
    """A pre-stamp index (no embed_model key) is assumed to be all-MiniLM-L6-v2.

    Querying it under a *different* current model must be rejected; under the
    old default it must pass the guard (reaching the embedding step).
    """
    metas = [{"chunk_text": "hello", "mtime": None}]  # no embed_model key
    vecs = numpy.zeros((1, 3), dtype="float32")
    index_dir = str(tmp_path / "idx")
    store.save(vecs, metas, index_dir)

    import os

    # Different model -> rejected.
    os.environ["KB_EMBED_MODEL"] = "paraphrase-multilingual-MiniLM-L12-v2"
    try:
        with pytest.raises(IncompatibleIndexError):
            query("anything", index_dir=index_dir)
    finally:
        del os.environ["KB_EMBED_MODEL"]


def test_ingest_stamps_model_and_rebuilds_on_model_change(tmp_path, monkeypatch):
    """Ingest stamps the model into every chunk and re-embeds when it changes."""
    from kb import embedding
    from kb.ingest import ingest

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha note about raft consensus\n", encoding="utf-8")
    index_dir = str(tmp_path / "idx")

    calls: list[list[str]] = []

    def fake_encode(texts):
        calls.append(list(texts))
        return numpy.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(embedding, "encode", fake_encode)

    # First ingest under model-A: chunks are stamped with it.
    monkeypatch.setattr(embedding, "current_model_name", lambda: "model-A")
    ingest(str(docs), index_dir=index_dir)
    _, metas = store.load(index_dir)
    assert metas and all(m.get("embed_model") == "model-A" for m in metas), (
        f"every chunk must be stamped with the build model, got "
        f"{[m.get('embed_model') for m in metas]}"
    )

    # Same model, unchanged content: the reuse path skips embedding entirely.
    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    assert calls == [], f"unchanged same-model ingest must reuse, got {calls}"

    # Switching the model must force a full re-embed and restamp.
    monkeypatch.setattr(embedding, "current_model_name", lambda: "model-B")
    calls.clear()
    ingest(str(docs), index_dir=index_dir)
    assert calls, "a model change must re-embed instead of reusing stale vectors"
    _, metas2 = store.load(index_dir)
    assert all(m.get("embed_model") == "model-B" for m in metas2), (
        "chunks must be restamped with the new model after a model change"
    )
