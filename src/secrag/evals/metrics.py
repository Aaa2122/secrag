"""Pure retrieval metrics — no I/O, fully unit-tested."""


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    if not relevant:
        raise ValueError("recall undefined without relevant items")
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr_at_k(relevant: set[int], retrieved: list[int], k: int = 10) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
