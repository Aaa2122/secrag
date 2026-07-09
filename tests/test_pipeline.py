import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from secrag.db import session_factory
from secrag.ingestion.pipeline import ingest_directory
from secrag.models import Chunk, Document
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "mini_10k.html"


async def _purge_acme():
    async with session_factory()() as session:
        await session.execute(text("DELETE FROM documents WHERE ticker = 'ACME'"))
        await session.commit()


@pytest.fixture
async def raw_dir(tmp_path):
    await _purge_acme()
    d = tmp_path / "ACME"
    d.mkdir()
    shutil.copy(FIXTURE, d / "FY2025_testacc001.html")
    (d / "FY2025_testacc001.json").write_text(
        json.dumps(
            {
                "ticker": "ACME",
                "cik": "0000000001",
                "company_name": "ACME Corp",
                "filing_type": "10-K",
                "fiscal_year": 2025,
                "accession_number": "test-acc-0001",
                "filing_date": "2025-03-01",
                "report_date": "2024-12-31",
                "source_url": "http://example.com/acme.htm",
            }
        ),
        encoding="utf-8",
    )
    yield tmp_path
    await _purge_acme()


async def test_ingest_is_idempotent_and_force_reingests(raw_dir):
    embedder = FakeEmbedder()

    stats = await ingest_directory(raw_dir, embedder=embedder)
    assert stats.documents_ingested == 1
    assert stats.chunks_written > 0
    assert stats.reports["ACME-FY2025"]["missing_core_items"] == []

    async with session_factory()() as session:
        doc = await session.scalar(select(Document).where(Document.ticker == "ACME"))
        assert doc is not None and doc.ingested_at is not None
        n = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
        assert n == stats.chunks_written
        sample = await session.scalar(select(Chunk).where(Chunk.document_id == doc.id).limit(1))
        assert sample.embedding is not None and len(sample.embedding) == 384
        assert sample.meta["item"] is not None or sample.meta["item_title"] == "Cover"

    again = await ingest_directory(raw_dir, embedder=embedder)
    assert again.documents_ingested == 0 and again.documents_skipped == 1

    forced = await ingest_directory(raw_dir, embedder=embedder, force=True)
    assert forced.documents_ingested == 1
