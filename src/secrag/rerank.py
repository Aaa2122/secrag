"""Local cross-encoder reranking behind a small interface.

The default MiniLM model is small enough for CPU deployment. Its quality and
latency are measured by the evals harness instead of being assumed.
"""

from functools import lru_cache
from typing import Protocol

from secrag.config import get_settings


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or get_settings().reranker_model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        return self._model.predict(pairs, batch_size=8, show_progress_bar=False).tolist()


@lru_cache
def get_reranker() -> Reranker:
    return CrossEncoderReranker()
