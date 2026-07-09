from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    ticker: str
    fiscal_year: int
    item: str | None
    item_title: str | None
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    timing_ms: dict[str, float]
