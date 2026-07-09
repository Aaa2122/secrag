"""Deterministic test doubles."""

import hashlib
import math


class FakeEmbedder:
    """Hash-based unit vectors; optional pinned vectors for exact texts."""

    def __init__(self, dim: int = 384, pinned: dict[str, list[float]] | None = None) -> None:
        self.dim = dim
        self._pinned = pinned or {}

    def _vec(self, text: str) -> list[float]:
        if text in self._pinned:
            return self._pinned[text]
        seed = hashlib.sha256(text.encode()).digest()
        raw = [(seed[i % len(seed)] - 127.5) for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
