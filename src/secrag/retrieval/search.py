"""Retrieval primitives: pgvector similarity, Postgres full-text, RRF hybrid."""

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from secrag.models import Chunk, Document

RRF_K = 60
CANDIDATE_K = 50
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|between|evol\w*)\b", re.I)
COMPARISON_WORD_RE = re.compile(
    r"\b(compare|comparison|comparing|compared|describe|describes|described|"
    r"evolve|evolved|evolution|both|between|versus|vs|how|what|does|do|did|"
    r"say|says|said|their|its|cited|impact|businesses|dependence|activity|most|"
    r"recent|annual|amounts?|"
    r"and|the|a|an|of|on|in|to|from|by)\b",
    re.I,
)
CORPORATE_SUFFIXES = {
    "co",
    "com",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "plc",
}
GENERIC_FIRST_WORDS = {"advanced"}


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


def _unique(values):
    return list(dict.fromkeys(values))


def _entity_aliases(ticker: str, company_name: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", company_name.lower())
    while words and words[-1] in CORPORATE_SUFFIXES:
        words.pop()
    aliases = [ticker.lower()]
    if words:
        full = " ".join(words)
        aliases.extend([full, "".join(words)])
        if len(words[0]) >= 4 and words[0] not in GENERIC_FIRST_WORDS:
            aliases.append(words[0])
    return _unique(aliases)


def _mentions_entity(query_text: str, ticker: str, company_name: str) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", query_text, re.I)
        for alias in _entity_aliases(ticker, company_name)
    )


def infer_query_scopes(
    query_text: str,
    filters: SearchFilters | None,
    entity_catalog: list[tuple[str, str]],
) -> list[SearchFilters]:
    """Split explicit comparisons by company, or by year for one company."""
    base = filters or SearchFilters()
    base_tickers = _unique(t.upper() for t in base.tickers)
    tickers = base_tickers or [
        ticker.upper()
        for ticker, name in entity_catalog
        if _mentions_entity(query_text, ticker, name)
    ]
    tickers = _unique(tickers)

    if len(tickers) > 1:
        return [
            SearchFilters(
                tickers=[ticker],
                fiscal_years=list(base.fiscal_years),
                items=list(base.items),
            )
            for ticker in tickers
        ]

    query_years = _unique(int(year) for year in YEAR_RE.findall(query_text))
    years = list(base.fiscal_years) if base.fiscal_years else query_years
    if len(years) > 1 and COMPARISON_RE.search(query_text):
        return [
            SearchFilters(tickers=list(tickers), fiscal_years=[year], items=list(base.items))
            for year in years
        ]
    return []


def comparison_topic(query_text: str, entity_catalog: list[tuple[str, str]]) -> str:
    """Remove entity names and comparison boilerplate for scoped full-text search."""
    topic = query_text.replace("’s", "").replace("'s", "")
    for ticker, name in entity_catalog:
        for alias in sorted(_entity_aliases(ticker, name), key=len, reverse=True):
            topic = re.sub(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", " ", topic, flags=re.I)
    topic = YEAR_RE.sub(" ", topic)
    topic = COMPARISON_WORD_RE.sub(" ", topic)
    topic = re.sub(r"[^a-z0-9$%.-]+", " ", topic.lower()).strip(" .-")
    return topic or query_text


def interleave_results(ranked_lists: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """Round-robin ranked lists while de-duplicating chunks."""
    merged: list[RetrievedChunk] = []
    seen: set[int] = set()
    max_len = max((len(ranked) for ranked in ranked_lists), default=0)
    for idx in range(max_len):
        for ranked in ranked_lists:
            if idx < len(ranked) and ranked[idx].chunk_id not in seen:
                merged.append(ranked[idx])
                seen.add(ranked[idx].chunk_id)
                if len(merged) == k:
                    return merged
    return merged


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


async def decomposed_hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    k: int = 10,
    filters: SearchFilters | None = None,
    candidate_k: int = CANDIDATE_K,
) -> tuple[list[RetrievedChunk], list[SearchFilters]]:
    """Hybrid search with balanced sub-retrieval for explicit comparisons."""
    catalog = list(
        (await session.execute(select(Document.ticker, Document.company_name).distinct())).all()
    )
    scopes = infer_query_scopes(query_text, filters, catalog)
    if not scopes:
        return (
            await hybrid_search(
                session,
                query_text,
                query_embedding,
                k=k,
                filters=filters,
                candidate_k=candidate_k,
            ),
            [],
        )

    topic = comparison_topic(query_text, catalog)
    ranked_lists = []
    for scope in scopes:
        scoped_query = " ".join([*scope.tickers, topic])
        ranked_lists.append(
            await hybrid_search(
                session,
                scoped_query,
                query_embedding,
                k=k,
                filters=scope,
                candidate_k=candidate_k,
            )
        )
    return interleave_results(ranked_lists, k), scopes


def _matches_scope(result: RetrievedChunk, scope: SearchFilters) -> bool:
    return (
        (not scope.tickers or result.ticker.upper() in {t.upper() for t in scope.tickers})
        and (not scope.fiscal_years or result.fiscal_year in scope.fiscal_years)
        and (not scope.items or (result.item or "").lower() in {i.lower() for i in scope.items})
    )


def rerank_results(
    reranker,
    query_text: str,
    results: list[RetrievedChunk],
    k: int,
    scopes: list[SearchFilters] | None = None,
) -> list[RetrievedChunk]:
    """Re-order candidates, reserving one final slot per comparison scope."""
    if not results:
        return []
    scores = reranker.score(query_text, [r.content for r in results])
    reranked = sorted(
        (replace(r, score=s) for r, s in zip(results, scores, strict=True)),
        key=lambda r: -r.score,
    )
    if not scopes or k < len(scopes):
        return reranked[:k]

    selected: list[RetrievedChunk] = []
    selected_ids: set[int] = set()
    for scope in scopes:
        match = next(
            (r for r in reranked if r.chunk_id not in selected_ids and _matches_scope(r, scope)),
            None,
        )
        if match:
            selected.append(match)
            selected_ids.add(match.chunk_id)
    selected.extend(r for r in reranked if r.chunk_id not in selected_ids)
    return sorted(selected[:k], key=lambda r: -r.score)
