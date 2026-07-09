import time
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from sqlalchemy.ext.asyncio import AsyncSession

from secrag import __version__
from secrag.api.schemas import SearchResponse, SearchResult
from secrag.db import session_factory
from secrag.embedding import Embedder, get_embedder
from secrag.retrieval.search import SearchFilters, vector_search

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
    mode: str = "vector",
    k: Annotated[int, Query(ge=1, le=50)] = 5,
    tickers: Annotated[list[str] | None, Query()] = None,
    fiscal_years: Annotated[list[int] | None, Query()] = None,
    items: Annotated[list[str] | None, Query()] = None,
) -> SearchResponse:
    filters = SearchFilters(
        tickers=tickers or [], fiscal_years=fiscal_years or [], items=items or []
    )
    t0 = time.perf_counter()
    query_embedding = embedder.embed_query(q)
    t1 = time.perf_counter()
    results = await vector_search(session, query_embedding, k=k, filters=filters)
    t2 = time.perf_counter()
    return SearchResponse(
        query=q,
        mode=mode,
        results=[SearchResult(**vars(r)) for r in results],
        timing_ms={
            "embed": round((t1 - t0) * 1000, 2),
            "search": round((t2 - t1) * 1000, 2),
        },
    )
