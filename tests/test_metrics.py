import pytest

from secrag.evals.metrics import mrr_at_k, percentile, recall_at_k


def test_recall_at_k():
    relevant = {1, 2, 3}
    retrieved = [9, 1, 8, 2, 7]
    assert recall_at_k(relevant, retrieved, k=1) == 0.0
    assert recall_at_k(relevant, retrieved, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(relevant, retrieved, k=5) == pytest.approx(2 / 3)
    assert recall_at_k({1}, [1], k=10) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(set(), [1], k=5)


def test_mrr_at_k():
    assert mrr_at_k({5}, [5, 1, 2], k=10) == 1.0
    assert mrr_at_k({5}, [1, 5, 2], k=10) == 0.5
    assert mrr_at_k({5}, [1, 2, 3], k=10) == 0.0
    assert mrr_at_k({5}, [1, 2, 5], k=2) == 0.0  # outside cutoff


def test_percentile():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile([7.0], 50) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
