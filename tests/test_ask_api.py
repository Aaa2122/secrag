import httpx
import pytest
from sqlalchemy import text

from secrag.api.main import app, embedder_dep
from secrag.db import session_factory
from secrag.models import Chunk, Document
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

V = [1.0] + [0.0] * 383


@pytest.fixture
async def seeded():
    async def purge():
        async with session_factory()() as session:
            await session.execute(text("DELETE FROM documents WHERE ticker = 'ASK1'"))
            await session.commit()

    await purge()
    async with session_factory()() as session:
        d = Document(
            ticker="ASK1",
            company_name="Ask Co",
            cik="8",
            fiscal_year=2025,
            accession_number="ask-1",
            source_url="http://x",
        )
        session.add(d)
        await session.flush()
        session.add(
            Chunk(
                document_id=d.id,
                chunk_index=0,
                content="Askco revenue was $12.34 billion in 2025.",
                embedding=V,
                meta={"item": "7", "item_title": "MD&A"},
            )
        )
        await session.commit()
    yield
    await purge()


async def _fake_generate(question, chunks):
    yield {"type": "token", "text": "Revenue was "}
    yield {"type": "token", "text": "$12.34 billion [1]."}
    yield {
        "type": "done",
        "model": "fake",
        "stop_reason": "end_turn",
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.00075,
    }


async def test_ask_streams_sources_tokens_done(seeded, monkeypatch):
    from secrag import generation

    monkeypatch.setattr(generation, "generate_answer", _fake_generate)
    app.dependency_overrides[embedder_dep] = lambda: FakeEmbedder(pinned={"askco revenue": V})
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.get(
                "/ask",
                params={"q": "askco revenue", "k": 1, "rerank": "false", "tickers": ["ASK1"]},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: sources" in body
    assert "ASK1 10-K FY2025, Item 7" in body
    assert "event: token" in body
    assert "$12.34 billion [1]." in body
    assert "event: done" in body
    assert '"cost_usd": 0.00075' in body
