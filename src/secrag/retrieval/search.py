"""Retrieval primitives: pgvector similarity, Postgres full-text, RRF hybrid."""

from collections import defaultdict
from dataclasses import dataclass, field, replace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from secrag.models import Chunk, Document

RRF_K = 60
CANDIDATE_K = 50


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


async def fulltext_search(
    session: AsyncSession,
    query_text: str,
    k: int = CANDIDATE_K,
    filters: SearchFilters | None = None,
) -> list[RetrievedChunk]:
    filters = filters or SearchFilters()
    tsq = func.websearch_to_tsquery("english", query_text)
    rank = func.ts_rank_cd(Chunk.tsv, tsq).label("rank")
    stmt = (
        select(Chunk, Document, rank)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tsv.op("@@")(tsq))
        .order_by(rank.desc(), Chunk.id)
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
            score=float(r),
        )
        for chunk, doc, r in rows
    ]


def rrf_fuse(ranked_lists: list[list[int]], k_rrf: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over lists of 1/(k + rank)."""
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for idx, chunk_id in enumerate(ranked):
            scores[chunk_id] += 1.0 / (k_rrf + idx + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    k: int = 10,
    filters: SearchFilters | None = None,
    candidate_k: int = CANDIDATE_K,
) -> list[RetrievedChunk]:
    vec = await vector_search(session, query_embedding, k=candidate_k, filters=filters)
    fts = await fulltext_search(session, query_text, k=candidate_k, filters=filters)
    by_id = {r.chunk_id: r for r in [*vec, *fts]}
    fused = rrf_fuse([[r.chunk_id for r in vec], [r.chunk_id for r in fts]])
    return [replace(by_id[cid], score=score) for cid, score in fused[:k]]


def rerank_results(
    reranker, query_text: str, results: list[RetrievedChunk], k: int
) -> list[RetrievedChunk]:
    """Re-order retrieval candidates with a cross-encoder; keep the top k."""
    if not results:
        return []
    scores = reranker.score(query_text, [r.content for r in results])
    reranked = sorted(
        (replace(r, score=s) for r, s in zip(results, scores, strict=True)),
        key=lambda r: -r.score,
    )
    return reranked[:k]
