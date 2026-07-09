"""Cross-encoder reranking behind a small interface.

Local bge-reranker-v2-m3 (CPU): $0 per query; the latency cost is measured
honestly by the evals harness instead of being hidden.
"""

from functools import lru_cache
from typing import Protocol

from secrag.config import get_settings


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class BgeReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or get_settings().reranker_model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        return self._model.predict(pairs, batch_size=8, show_progress_bar=False).tolist()


@lru_cache
def get_reranker() -> Reranker:
    return BgeReranker()
