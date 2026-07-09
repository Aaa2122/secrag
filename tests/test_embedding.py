import math

import pytest

from secrag.embedding import BGE_QUERY_PREFIX, get_embedder


def test_factory_rejects_unknown_provider(monkeypatch):
    from secrag import config, embedding

    monkeypatch.setenv("EMBEDDING_PROVIDER", "nope")
    config.get_settings.cache_clear()
    embedding.get_embedder.cache_clear()
    try:
        with pytest.raises(ValueError, match="unknown embedding provider"):
            get_embedder()
    finally:
        config.get_settings.cache_clear()
        embedding.get_embedder.cache_clear()


def test_bge_query_prefix_is_the_documented_instruction():
    assert BGE_QUERY_PREFIX.startswith("Represent this sentence")


@pytest.mark.integration
def test_local_embedder_real_model():
    from secrag.embedding import LocalEmbedder

    emb = LocalEmbedder()
    assert emb.dim == 384

    vecs = emb.embed_passages(["NVIDIA revenue grew.", "Apple sells iPhones."])
    assert len(vecs) == 2 and len(vecs[0]) == 384
    assert math.isclose(sum(x * x for x in vecs[0]), 1.0, rel_tol=1e-3)  # normalized

    q = emb.embed_query("NVIDIA revenue grew.")
    assert q != vecs[0]  # query instruction makes it asymmetric

    # sanity: related sentence closer to its passage than the unrelated one
    dot_related = sum(a * b for a, b in zip(q, vecs[0], strict=True))
    dot_unrelated = sum(a * b for a, b in zip(q, vecs[1], strict=True))
    assert dot_related > dot_unrelated
