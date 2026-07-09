import json
import time
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from secrag import __version__, generation
from secrag.api.schemas import SearchResponse, SearchResult
from secrag.config import get_settings
from secrag.db import session_factory
from secrag.embedding import Embedder, get_embedder
from secrag.rerank import get_reranker
from secrag.retrieval.search import SearchFilters, hybrid_search, rerank_results, vector_search

app = FastAPI(title="secrag", version=__version__)


async def get_session():
    async with session_factory()() as session:
        yield session


def embedder_dep() -> Embedder:
    return get_embedder()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=2)],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
    mode: Literal["vector", "hybrid"] = "hybrid",
    k: Annotated[int, Query(ge=1, le=50)] = 5,
    rerank: bool = False,
    tickers: Annotated[list[str] | None, Query()] = None,
    fiscal_years: Annotated[list[int] | None, Query()] = None,
    items: Annotated[list[str] | None, Query()] = None,
) -> SearchResponse:
    filters = SearchFilters(
        tickers=tickers or [], fiscal_years=fiscal_years or [], items=items or []
    )
    results, timing = await _retrieve(session, embedder, q, mode, k, rerank, filters)
    return SearchResponse(
        query=q,
        mode=mode,
        results=[SearchResult(**vars(r)) for r in results],
        timing_ms=timing,
    )


async def _retrieve(session, embedder, q, mode, k, rerank, filters):
    fetch_k = get_settings().rerank_candidates if rerank else k
    t0 = time.perf_counter()
    query_embedding = embedder.embed_query(q)
    t1 = time.perf_counter()
    if mode == "hybrid":
        results = await hybrid_search(session, q, query_embedding, k=fetch_k, filters=filters)
    else:
        results = await vector_search(session, query_embedding, k=fetch_k, filters=filters)
    t2 = time.perf_counter()
    timing = {
        "embed": round((t1 - t0) * 1000, 2),
        "search": round((t2 - t1) * 1000, 2),
    }
    if rerank:
        # get_reranker is a cached singleton, loaded on first rerank=true request;
        # tests monkeypatch it to avoid the multi-GB model download.
        results = rerank_results(get_reranker(), q, results, k=k)
        timing["rerank"] = round((time.perf_counter() - t2) * 1000, 2)
    return results, timing


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/ask")
async def ask(
    q: Annotated[str, Query(min_length=2)],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
    k: Annotated[int, Query(ge=1, le=10)] = 5,
    rerank: bool = True,
    tickers: Annotated[list[str] | None, Query()] = None,
    fiscal_years: Annotated[list[int] | None, Query()] = None,
    items: Annotated[list[str] | None, Query()] = None,
) -> StreamingResponse:
    """Grounded answer as SSE: `sources` event, `token` events, then `done`."""
    filters = SearchFilters(
        tickers=tickers or [], fiscal_years=fiscal_years or [], items=items or []
    )
    results, timing = await _retrieve(session, embedder, q, "hybrid", k, rerank, filters)

    async def stream():
        yield _sse(
            "sources",
            {
                "sources": [
                    {
                        "n": i,
                        "chunk_id": r.chunk_id,
                        "label": generation.source_label(r),
                        "ticker": r.ticker,
                        "fiscal_year": r.fiscal_year,
                        "item": r.item,
                        "content": r.content,
                    }
                    for i, r in enumerate(results, 1)
                ],
                "timing_ms": timing,
            },
        )
        # generation.generate_answer is monkeypatched in tests (no API key needed).
        async for event in generation.generate_answer(q, results):
            if event["type"] == "token":
                yield _sse("token", {"text": event["text"]})
            else:
                yield _sse("done", event)

    return StreamingResponse(stream(), media_type="text/event-stream")
