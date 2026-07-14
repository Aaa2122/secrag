from secrag.retrieval.search import (
    RetrievedChunk,
    SearchFilters,
    comparison_topic,
    infer_query_scopes,
    interleave_results,
    rerank_results,
)

CATALOG = [
    ("AAPL", "Apple Inc."),
    ("AMZN", "AMAZON COM INC"),
    ("AMD", "ADVANCED MICRO DEVICES INC"),
    ("GOOGL", "Alphabet Inc."),
    ("JPM", "JPMORGAN CHASE & CO"),
    ("META", "Meta Platforms, Inc."),
    ("MSFT", "MICROSOFT CORP"),
    ("NVDA", "NVIDIA CORP"),
    ("TSLA", "Tesla, Inc."),
    ("XOM", "EXXON MOBIL CORP"),
]


def _chunk(cid: int, ticker: str, year: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(cid, cid, ticker, year, "1a", "Risk Factors", f"chunk {cid}", score)


class PinnedReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, passages):
        return [self.scores[p] for p in passages]


def test_infer_query_scopes_from_company_names_and_compact_name():
    scopes = infer_query_scopes(
        "Compare Apple, Amazon and JPMorganChase supplier risks.", None, CATALOG
    )
    assert [scope.tickers for scope in scopes] == [["AAPL"], ["AMZN"], ["JPM"]]


def test_explicit_ticker_filters_take_precedence():
    scopes = infer_query_scopes(
        "Compare Apple and Amazon suppliers.", SearchFilters(tickers=["AAPL"]), CATALOG
    )
    assert scopes == []


def test_infer_year_scopes_only_for_explicit_comparison():
    comparison = infer_query_scopes(
        "How did Tesla revenue evolve between 2024 and 2025?", None, CATALOG
    )
    assert comparison == [
        SearchFilters(tickers=["TSLA"], fiscal_years=[2024]),
        SearchFilters(tickers=["TSLA"], fiscal_years=[2025]),
    ]
    assert infer_query_scopes("What was Meta revenue in 2025, 2024 and 2023?", None, CATALOG) == []


def test_comparison_topic_removes_entities_years_and_boilerplate():
    topic = comparison_topic(
        "Compare NVIDIA's and AMD's most recent annual stock repurchase amounts in 2025.",
        CATALOG,
    )
    assert topic == "stock repurchase"
    assert (
        comparison_topic(
            "Compare how Apple and Amazon describe their dependence on suppliers.", CATALOG
        )
        == "suppliers"
    )
    assert (
        comparison_topic(
            "Compare Exxon Mobil's and NVIDIA's most recent share repurchase activity.", CATALOG
        )
        == "share repurchase"
    )


def test_interleave_results_balances_and_deduplicates():
    one = [_chunk(1, "AAPL", 2025), _chunk(2, "AAPL", 2024)]
    two = [_chunk(3, "AMZN", 2025), _chunk(2, "AAPL", 2024)]
    assert [r.chunk_id for r in interleave_results([one, two], k=4)] == [1, 3, 2]


def test_reranker_reserves_one_result_per_scope():
    results = [
        _chunk(1, "AAPL", 2025),
        _chunk(2, "AAPL", 2024),
        _chunk(3, "AMZN", 2025),
    ]
    reranker = PinnedReranker({"chunk 1": 0.9, "chunk 2": 0.8, "chunk 3": 0.1})
    scopes = [SearchFilters(tickers=["AAPL"]), SearchFilters(tickers=["AMZN"])]
    top = rerank_results(reranker, "compare", results, k=2, scopes=scopes)
    assert {r.ticker for r in top} == {"AAPL", "AMZN"}
