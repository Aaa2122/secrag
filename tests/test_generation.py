from secrag.generation import (
    REFUSAL_SENTENCE,
    SYSTEM_PROMPT,
    build_user_prompt,
    estimate_cost_usd,
    source_label,
)
from secrag.retrieval.search import RetrievedChunk


def _chunk(cid: int, ticker: str = "NVDA", item: str | None = "1a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=1,
        ticker=ticker,
        fiscal_year=2026,
        item=item,
        item_title="Risk Factors",
        content=f"content of chunk {cid}",
        score=0.9,
    )


def test_system_prompt_demands_citations_and_refusal():
    assert "[n]" in SYSTEM_PROMPT
    assert REFUSAL_SENTENCE in SYSTEM_PROMPT
    assert "no rounding" in SYSTEM_PROMPT


def test_source_label_formats_citation_target():
    assert source_label(_chunk(1)) == "NVDA 10-K FY2026, Item 1A"
    assert source_label(_chunk(1, item=None)) == "NVDA 10-K FY2026, Cover"


def test_build_user_prompt_numbers_sources_in_order():
    prompt = build_user_prompt("What risks?", [_chunk(10), _chunk(20, ticker="AMD")])
    assert prompt.index("[1] (NVDA 10-K FY2026, Item 1A)") < prompt.index(
        "[2] (AMD 10-K FY2026, Item 1A)"
    )
    assert "content of chunk 10" in prompt
    assert prompt.rstrip().endswith("Question: What risks?")


def test_estimate_cost_usd():
    # claude-opus-4-8: $5/MTok in, $25/MTok out
    assert estimate_cost_usd("claude-opus-4-8", 1_000_000, 0) == 5.0
    assert estimate_cost_usd("claude-opus-4-8", 0, 1_000_000) == 25.0
    assert estimate_cost_usd("claude-opus-4-8", 10_000, 500) == 0.0625
    assert estimate_cost_usd("unknown-model", 1000, 1000) is None
