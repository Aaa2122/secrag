"""Ingest downloaded filings (html + sidecar json) into Postgres.

Idempotent: a document whose accession_number already has chunks is skipped
unless --force. Embedding cost is $0.00 (local model) — logged anyway because
cost-per-stage is part of this project's evals story.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from secrag.db import session_factory
from secrag.embedding import Embedder, get_embedder
from secrag.ingestion.chunker import chunk_sections
from secrag.ingestion.parse10k import parse_10k, parse_report
from secrag.models import Chunk, Document

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    embed_seconds: float = 0.0
    reports: dict[str, dict] = field(default_factory=dict)


async def _ingest_filing(
    session: AsyncSession, html_path: Path, meta: dict, embedder: Embedder, force: bool
) -> tuple[dict, int, float] | None:
    doc = await session.scalar(
        select(Document).where(Document.accession_number == meta["accession_number"])
    )
    if doc is not None:
        n_existing = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
        if n_existing and not force:
            return None
        await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    else:
        doc = Document(
            ticker=meta["ticker"],
            company_name=meta["company_name"],
            cik=meta["cik"],
            filing_type=meta["filing_type"],
            fiscal_year=meta["fiscal_year"],
            accession_number=meta["accession_number"],
            source_url=meta["source_url"],
            filed_at=date.fromisoformat(meta["filing_date"]),
        )
        session.add(doc)
        await session.flush()

    sections = parse_10k(html_path.read_text(encoding="utf-8"))
    report = parse_report(sections)
    drafts = chunk_sections(sections, ticker=meta["ticker"], fiscal_year=meta["fiscal_year"])

    t0 = time.perf_counter()
    vectors = embedder.embed_passages([d.content for d in drafts])
    embed_s = time.perf_counter() - t0

    session.add_all(
        Chunk(
            document_id=doc.id,
            chunk_index=i,
            content=d.content,
            token_count=d.token_count,
            embedding=v,
            meta=d.meta,
        )
        for i, (d, v) in enumerate(zip(drafts, vectors, strict=True))
    )
    doc.ingested_at = datetime.now(UTC)
    await session.commit()
    return report, len(drafts), embed_s


async def ingest_directory(
    raw_dir: Path, *, force: bool = False, embedder: Embedder | None = None
) -> IngestStats:
    embedder = embedder or get_embedder()
    stats = IngestStats()
    factory = session_factory()
    for html_path in sorted(raw_dir.glob("*/*.html")):
        meta = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
        key = f"{meta['ticker']}-FY{meta['fiscal_year']}"
        async with factory() as session:
            result = await _ingest_filing(session, html_path, meta, embedder, force)
        if result is None:
            stats.documents_skipped += 1
            log.info("skip (already ingested): %s", key)
            continue
        report, n_chunks, embed_s = result
        stats.documents_ingested += 1
        stats.chunks_written += n_chunks
        stats.embed_seconds += embed_s
        stats.reports[key] = report
        log.info(
            "ingested %s: %d chunks, embed %.1fs, missing_core=%s",
            key,
            n_chunks,
            embed_s,
            report["missing_core_items"],
        )
    log.info(
        "done: %d ingested, %d skipped, %d chunks, embed %.1fs total, embedding cost $0.00",
        stats.documents_ingested,
        stats.documents_skipped,
        stats.chunks_written,
        stats.embed_seconds,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest downloaded 10-K filings into Postgres")
    parser.add_argument("raw_dir", type=Path, nargs="?", default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="re-ingest existing documents")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(ingest_directory(args.raw_dir, force=args.force))


if __name__ == "__main__":
    main()
