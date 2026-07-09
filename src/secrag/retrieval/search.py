"""Retrieval primitives: vector search over pgvector (hybrid lands in Jalon 4)."""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from secrag.models import Chunk, Document


@dataclass(frozen=True)
class SearchFilters:
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    ticker: str
    fiscal_year: int
    item: str | None
    item_title: str | None
    content: str
    score: float  # higher is better (cosine similarity for vector mode)


def _apply_filters(stmt, filters: SearchFilters):
    if filters.tickers:
        stmt = stmt.where(Document.ticker.in_([t.upper() for t in filters.tickers]))
    if filters.fiscal_years:
        stmt = stmt.where(Document.fiscal_year.in_(filters.fiscal_years))
    if filters.items:
        stmt = stmt.where(Chunk.meta["item"].astext.in_([i.lower() for i in filters.items]))
    return stmt


async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    k: int = 10,
    filters: SearchFilters | None = None,
) -> list[RetrievedChunk]:
    filters = filters or SearchFilters()
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(Chunk, Document, distance)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    stmt = _apply_filters(stmt, filters)
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=doc.id,
            ticker=doc.ticker,
            fiscal_year=doc.fiscal_year,
            item=chunk.meta.get("item"),
            item_title=chunk.meta.get("item_title"),
            content=chunk.content,
            score=1.0 - dist,  # cosine distance -> similarity
        )
        for chunk, doc, dist in rows
    ]
