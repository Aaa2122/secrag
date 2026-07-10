"""Embedding providers behind a small interface.

Default is a local sentence-transformers model: the retrieval stack then costs
$0 per query and needs no API key. Swapping providers means changing settings,
re-ingesting the corpus, and (if dims change) a schema migration.
"""

from functools import lru_cache
from typing import Protocol

from secrag.config import get_settings

# bge-small-en-v1.5 is asymmetric: queries need this instruction, passages don't.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        # Lazy import: keeps API/test startup fast when the embedder is unused.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or get_settings().embedding_model)
        self.dim = self._model.get_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            [BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


@lru_cache
def get_embedder() -> Embedder:
    provider = get_settings().embedding_provider
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(f"unknown embedding provider: {provider!r}")
