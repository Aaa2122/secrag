import httpx
import pytest
from sqlalchemy import text

from secrag.api.main import app, embedder_dep
from secrag.db import session_factory
from secrag.models import Chunk, Document
from secrag.retrieval.search import SearchFilters, vector_search
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

# Pin vectors so ranking is fully deterministic: the query matches doc A's chunk.
V_MATCH = [1.0] + [0.0] * 383
V_NEAR = [0.9] + [(0.19**0.5 / 383**0.5)] * 383  # normalized-ish, close to V_MATCH
V_FAR = [0.0] * 383 + [1.0]

EMBEDDER = FakeEmbedder(
    pinned={
        "anvil supply risk": V_MATCH,
        "PIN-A": V_MATCH,
        "PIN-B": V_NEAR,
        "PIN-C": V_FAR,
    }
)


async def _purge():
    async with session_factory()() as session:
        await session.execute(text("DELETE FROM documents WHERE ticker IN ('SRCH1','SRCH2')"))
        await session.commit()


@pytest.fixture
async def seeded():
    await _purge()
    async with session_factory()() as session:
        d1 = Document(
            ticker="SRCH1",
            company_name="Search One",
            cik="1",
            fiscal_year=2025,
            accession_number="srch-1",
            source_url="http://x",
        )
        d2 = Document(
            ticker="SRCH2",
            company_name="Search Two",
            cik="2",
            fiscal_year=2024,
            accession_number="srch-2",
            source_url="http://x",
        )
        session.add_all([d1, d2])
        await session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=d1.id,
                    chunk_index=0,
                    content="PIN-A anvils",
                    embedding=V_MATCH,
                    meta={"item": "1a", "item_title": "Risk Factors"},
                ),
                Chunk(
                    document_id=d1.id,
                    chunk_index=1,
                    content="PIN-B rockets",
                    embedding=V_NEAR,
                    meta={"item": "7", "item_title": "MD&A"},
                ),
                Chunk(
                    document_id=d2.id,
                    chunk_index=0,
                    content="PIN-C unrelated",
                    embedding=V_FAR,
                    meta={"item": "1a", "item_title": "Risk Factors"},
                ),
            ]
        )
        await session.commit()
    yield
    await _purge()


SCOPE = SearchFilters(tickers=["SRCH1", "SRCH2"])  # the DB also holds the real corpus


async def test_vector_search_ranks_by_similarity(seeded):
    async with session_factory()() as session:
        results = await vector_search(session, V_MATCH, k=3, filters=SCOPE)
    assert [r.content for r in results[:2]] == ["PIN-A anvils", "PIN-B rockets"]
    assert results[0].score > results[1].score > 0


async def test_filters_restrict_scope(seeded):
    async with session_factory()() as session:
        only_2024 = await vector_search(
            session,
            V_MATCH,
            k=5,
            filters=SearchFilters(tickers=["SRCH1", "SRCH2"], fiscal_years=[2024]),
        )
        only_7 = await vector_search(
            session,
            V_MATCH,
            k=5,
            filters=SearchFilters(tickers=["srch1"], items=["7"]),
        )
    assert [r.ticker for r in only_2024] == ["SRCH2"]
    assert len(only_7) == 1 and only_7[0].item == "7"


async def test_search_endpoint_returns_results_and_timings(seeded):
    app.dependency_overrides[embedder_dep] = lambda: EMBEDDER
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.get(
                "/search",
                params={"q": "anvil supply risk", "k": 2, "tickers": ["SRCH1", "SRCH2"]},
            )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["content"] == "PIN-A anvils"
    assert body["results"][0]["item"] == "1a"
    assert set(body["timing_ms"]) == {"embed", "search"}
