import pytest
from sqlalchemy import text

from secrag.db import get_engine, session_factory
from secrag.models import Chunk, Document

pytestmark = pytest.mark.integration


async def test_schema_tables_and_indexes_exist():
    engine = get_engine()
    async with engine.connect() as conn:
        tables = (
            (await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")))
            .scalars()
            .all()
        )
        assert {"documents", "chunks"} <= set(tables)

        indexes = (
            (await conn.execute(text("SELECT indexname FROM pg_indexes WHERE tablename='chunks'")))
            .scalars()
            .all()
        )
        assert "ix_chunks_embedding_hnsw" in indexes
        assert "ix_chunks_tsv" in indexes


async def test_chunk_roundtrip_with_vector_and_tsv():
    async with session_factory()() as session:
        doc = Document(
            ticker="TEST",
            company_name="Test Co",
            cik="0000000000",
            fiscal_year=2099,
            accession_number="test-acc-1",
            source_url="http://example.com",
        )
        session.add(doc)
        await session.flush()
        session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=0,
                content="Revenue grew twelve percent",
                embedding=[0.1] * 384,
                meta={"item": "7"},
            )
        )
        await session.commit()

    async with session_factory()() as session:
        row = (
            await session.execute(
                text(
                    "SELECT tsv IS NOT NULL, embedding IS NOT NULL FROM chunks "
                    "WHERE content = 'Revenue grew twelve percent'"
                )
            )
        ).one()
        assert row == (True, True)
        await session.execute(text("DELETE FROM documents WHERE ticker = 'TEST'"))
        await session.commit()
