"""Retrieval evals runner: versioned JSON results + one-line changelog.

Usage: uv run python -m secrag.evals.run --mode vector --label vector-baseline
"""

import argparse
import asyncio
import json
import logging
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from secrag.config import get_settings
from secrag.db import session_factory
from secrag.embedding import get_embedder
from secrag.evals.golden import GOLDEN_PATH, load_golden, resolve_refs
from secrag.evals.metrics import mrr_at_k, percentile, recall_at_k
from secrag.retrieval.search import hybrid_search, vector_search

log = logging.getLogger(__name__)

RESULTS_DIR = Path("evals/results")
CHANGELOG = Path("evals/CHANGELOG.md")
K_VALUES = (1, 3, 5, 10)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


async def run_retrieval_eval(mode: str, label: str) -> dict:
    questions = [q for q in load_golden() if q.answerable]
    embedder = get_embedder()
    factory = session_factory()

    async with factory() as session:
        relevant_by_q = await resolve_refs(session, questions)

    per_question: list[dict] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []

    async with factory() as session:
        for q in questions:
            t0 = time.perf_counter()
            qvec = embedder.embed_query(q.question)
            t1 = time.perf_counter()
            if mode == "vector":
                results = await vector_search(session, qvec, k=max(K_VALUES))
            elif mode == "hybrid":
                results = await hybrid_search(session, q.question, qvec, k=max(K_VALUES))
            else:
                raise ValueError(f"unknown mode: {mode!r}")
            t2 = time.perf_counter()

            retrieved = [r.chunk_id for r in results]
            relevant = relevant_by_q[q.id]
            embed_ms.append((t1 - t0) * 1000)
            search_ms.append((t2 - t1) * 1000)
            per_question.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "recall": {k: recall_at_k(relevant, retrieved, k) for k in K_VALUES},
                    "mrr": mrr_at_k(relevant, retrieved, k=10),
                }
            )

    categories = sorted({p["category"] for p in per_question})

    def agg(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            **{f"recall@{k}": statistics.mean(r["recall"][k] for r in rows) for k in K_VALUES},
            "mrr@10": statistics.mean(r["mrr"] for r in rows),
        }

    total_ms = [e + s for e, s in zip(embed_ms, search_ms, strict=True)]
    result = {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "config": {
            "mode": mode,
            "embedder": get_settings().embedding_model,
            "reranker": None,
            "k_values": list(K_VALUES),
        },
        "aggregate": agg(per_question),
        "per_category": {
            c: agg([p for p in per_question if p["category"] == c]) for c in categories
        },
        "latency_ms": {
            "embed_p50": round(percentile(embed_ms, 50), 1),
            "embed_p95": round(percentile(embed_ms, 95), 1),
            "search_p50": round(percentile(search_ms, 50), 1),
            "search_p95": round(percentile(search_ms, 95), 1),
            "total_p50": round(percentile(total_ms, 50), 1),
            "total_p95": round(percentile(total_ms, 95), 1),
        },
        "cost_per_query_usd": 0.0,  # local embedder, no paid API in the retrieval path
        "per_question": per_question,
    }
    return result


def write_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date = result["created_at"][:10]
    out = RESULTS_DIR / f"{date}-{result['label']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    agg = result["aggregate"]
    line = (
        f"- `{date}` **{result['label']}** ({result['git_rev']}, mode={result['config']['mode']}):"
        f" recall@5 {agg['recall@5']:.3f} · recall@10 {agg['recall@10']:.3f}"
        f" · MRR@10 {agg['mrr@10']:.3f} · p95 {result['latency_ms']['total_p95']:.0f}ms"
        f" · ${result['cost_per_query_usd']:.4f}/query\n"
    )
    if not CHANGELOG.exists():
        CHANGELOG.write_text("# Evals changelog\n\n", encoding="utf-8")
    CHANGELOG.write_text(CHANGELOG.read_text(encoding="utf-8") + line, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evals against the golden dataset")
    parser.add_argument("--mode", default="vector", choices=["vector", "hybrid"])
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not GOLDEN_PATH.exists():
        raise SystemExit(f"golden dataset not found: {GOLDEN_PATH}")
    result = asyncio.run(run_retrieval_eval(args.mode, args.label))
    out = write_result(result)
    agg = result["aggregate"]
    log.info("wrote %s", out)
    log.info(
        "n=%d recall@5=%.3f recall@10=%.3f mrr@10=%.3f p95=%.0fms",
        agg["n"],
        agg["recall@5"],
        agg["recall@10"],
        agg["mrr@10"],
        result["latency_ms"]["total_p95"],
    )


if __name__ == "__main__":
    main()
