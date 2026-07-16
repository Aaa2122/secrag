import pytest

from secrag.retrieval.search import RetrievedChunk, rerank_results


class PinnedReranker:
    """Scores passages by a pinned mapping (default 0.0)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [self._scores.get(p, 0.0) for p in passages]


def _chunk(cid: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=1,
        ticker="T",
        fiscal_year=2025,
        item="1",
        item_title="Business",
        content=content,
        score=0.5,
    )


def test_rerank_reorders_and_truncates():
    results = [_chunk(1, "weak"), _chunk(2, "strong"), _chunk(3, "medium")]
    reranker = PinnedReranker({"weak": 0.1, "strong": 0.9, "medium": 0.5})
    top = rerank_results(reranker, "q", results, k=2)
    assert [r.chunk_id for r in top] == [2, 3]
    assert top[0].score == 0.9


def test_rerank_empty_input():
    assert rerank_results(PinnedReranker({}), "q", [], k=5) == []


@pytest.mark.heavy
def test_cross_encoder_reranker_scores_relevant_passage_higher():
    from secrag.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scores = reranker.score(
        "What was NVIDIA's data center revenue growth?",
        [
            "NVIDIA Data Center computing revenue grew 162% driven by Hopper demand.",
            "The company leases office space in Santa Clara, California.",
        ],
    )
    assert scores[0] > scores[1]
