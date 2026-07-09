import pytest
from sqlalchemy import text

from secrag.db import session_factory
from secrag.models import Chunk, Document
from secrag.retrieval.search import SearchFilters, fulltext_search, hybrid_search, rrf_fuse

V_QUERY = [1.0] + [0.0] * 383
V_CLOSE = [0.98] + [0.0396] * 383
V_FAR = [0.0] * 383 + [1.0]

SCOPE = SearchFilters(tickers=["HYB1"])


def test_rrf_fuse_rewards_presence_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [2, 4, 5]], k_rrf=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == 2  # appears in both lists
    scores = dict(fused)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 61)


def test_rrf_fuse_deterministic_tie_break():
    assert rrf_fuse([[7], [9]]) == [(7, pytest.approx(1 / 61)), (9, pytest.approx(1 / 61))][:2]


@pytest.fixture
async def seeded():
    async def purge():
        async with session_factory()() as session:
            await session.execute(text("DELETE FROM documents WHERE ticker = 'HYB1'"))
            await session.commit()

    await purge()
    async with session_factory()() as session:
        d = Document(
            ticker="HYB1",
            company_name="Hybrid Co",
            cik="9",
            fiscal_year=2025,
            accession_number="hyb-1",
            source_url="http://x",
        )
        session.add(d)
        await session.flush()
        session.add_all(
            [
                Chunk(
                    document_id=d.id,
                    chunk_index=0,
                    content="Our zorblatt reactor uses exotic fuel.",
                    embedding=V_FAR,
                    meta={"item": "1", "item_title": "Business"},
                ),
                Chunk(
                    document_id=d.id,
                    chunk_index=1,
                    content="Generic business commentary about growth.",
                    embedding=V_CLOSE,
                    meta={"item": "7", "item_title": "MD&A"},
                ),
            ]
        )
        await session.commit()
    yield
    await purge()


@pytest.mark.integration
async def test_fulltext_finds_exact_rare_term(seeded):
    async with session_factory()() as session:
        results = await fulltext_search(session, "zorblatt reactor", filters=SCOPE)
    assert len(results) == 1
    assert "zorblatt" in results[0].content


@pytest.mark.integration
async def test_hybrid_surfaces_exact_term_missed_by_vector(seeded):
    async with session_factory()() as session:
        top = await hybrid_search(session, "zorblatt reactor", V_QUERY, k=2, filters=SCOPE)
    # Vector alone ranks the generic chunk first (V_CLOSE); full-text pulls the
    # zorblatt chunk in; RRF puts the exact-term chunk on top of the fusion.
    assert {r.content.split()[1] for r in top} == {"zorblatt", "business"}
    assert "zorblatt" in top[0].content
