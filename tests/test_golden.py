import json
from pathlib import Path

import pytest

from secrag.evals.golden import (
    GOLDEN_PATH,
    GoldenValidationError,
    load_golden,
    resolve_refs,
)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


GOOD = {
    "id": "q1",
    "category": "qualitative",
    "question": "What supply chain risks does ACME face?",
    "expected_answer": "Dependence on a single anvil supplier.",
    "refs": [
        {"ticker": "ACME", "fiscal_year": 2025, "quote": "we depend on a single anvil supplier"}
    ],
}


def test_load_valid_golden(tmp_path):
    qs = load_golden(_write(tmp_path, [GOOD]))
    assert len(qs) == 1 and qs[0].answerable and qs[0].refs[0].ticker == "ACME"


def test_unanswerable_must_have_no_refs(tmp_path):
    bad = {**GOOD, "id": "q2", "category": "unanswerable"}
    with pytest.raises(GoldenValidationError, match="must have no refs"):
        load_golden(_write(tmp_path, [bad]))


def test_answerable_needs_refs(tmp_path):
    bad = {**GOOD, "id": "q3", "refs": []}
    with pytest.raises(GoldenValidationError, match="needs refs"):
        load_golden(_write(tmp_path, [bad]))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(GoldenValidationError, match="duplicate"):
        load_golden(_write(tmp_path, [GOOD, GOOD]))


def test_short_quotes_rejected(tmp_path):
    bad = {**GOOD, "id": "q4", "refs": [{"ticker": "ACME", "fiscal_year": 2025, "quote": "anvils"}]}
    with pytest.raises(GoldenValidationError, match="too short"):
        load_golden(_write(tmp_path, [bad]))


def test_unknown_category_rejected(tmp_path):
    bad = {**GOOD, "id": "q5", "category": "vibes"}
    with pytest.raises(GoldenValidationError, match="unknown category"):
        load_golden(_write(tmp_path, [bad]))


@pytest.mark.integration
async def test_committed_golden_dataset_fully_resolves():
    """Every quote in the committed golden dataset must match >=1 real chunk."""
    if not GOLDEN_PATH.exists():
        pytest.skip("golden dataset not yet authored")
    from secrag.db import session_factory

    questions = load_golden()
    assert len(questions) >= 40
    async with session_factory()() as session:
        resolved = await resolve_refs(session, [q for q in questions if q.answerable])
    assert all(ids for ids in resolved.values())
