"""Golden dataset: quote-anchored questions, resilient to re-chunking.

Each reference is a verbatim substring of some chunk's content, scoped to one
(ticker, fiscal_year) document. Resolution maps quotes to current chunk ids at
eval time, so goldens survive chunking changes; a quote that no longer matches
any chunk fails loudly instead of silently shrinking the dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from secrag.models import Chunk, Document

CATEGORIES = {"factual_numeric", "qualitative", "cross_doc", "unanswerable"}

GOLDEN_PATH = Path("evals/golden.jsonl")


@dataclass(frozen=True)
class Ref:
    ticker: str
    fiscal_year: int
    quote: str


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    category: str
    question: str
    expected_answer: str
    refs: tuple[Ref, ...]

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


class GoldenValidationError(ValueError):
    pass


class GoldenResolutionError(RuntimeError):
    pass


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenValidationError(f"line {lineno}: invalid JSON: {exc}") from exc
        q = GoldenQuestion(
            id=raw["id"],
            category=raw["category"],
            question=raw["question"],
            expected_answer=raw["expected_answer"],
            refs=tuple(Ref(**r) for r in raw.get("refs", [])),
        )
        if q.category not in CATEGORIES:
            raise GoldenValidationError(f"{q.id}: unknown category {q.category!r}")
        if q.id in seen_ids:
            raise GoldenValidationError(f"duplicate id {q.id}")
        if not q.question.strip() or not q.expected_answer.strip():
            raise GoldenValidationError(f"{q.id}: empty question or expected_answer")
        if q.answerable and not q.refs:
            raise GoldenValidationError(f"{q.id}: answerable question needs refs")
        if not q.answerable and q.refs:
            raise GoldenValidationError(f"{q.id}: unanswerable question must have no refs")
        for ref in q.refs:
            if len(ref.quote) < 20:
                raise GoldenValidationError(f"{q.id}: quote too short to be discriminative")
        seen_ids.add(q.id)
        questions.append(q)
    return questions


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def resolve_refs(
    session: AsyncSession, questions: list[GoldenQuestion]
) -> dict[str, set[int]]:
    """Map each answerable question id to the set of chunk ids its quotes live in."""
    resolved: dict[str, set[int]] = {}
    failures: list[str] = []
    for q in questions:
        if not q.answerable:
            continue
        ids: set[int] = set()
        for ref in q.refs:
            stmt = (
                select(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Document.ticker == ref.ticker.upper(),
                    Document.fiscal_year == ref.fiscal_year,
                    Chunk.content.like(f"%{_like_escape(ref.quote)}%", escape="\\"),
                )
            )
            matches = set((await session.execute(stmt)).scalars().all())
            if not matches:
                failures.append(f"{q.id}: quote not found in {ref.ticker} FY{ref.fiscal_year}")
            ids |= matches
        resolved[q.id] = ids
    if failures:
        raise GoldenResolutionError("; ".join(failures))
    return resolved
