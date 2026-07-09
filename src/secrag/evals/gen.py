"""Generation evals: faithfulness (LLM judge), figure provenance, citations,
refusal correctness. Checkpointed so an interrupted run never re-spends API
credits on questions already answered.

The headline claim this enables: **zero fabricated figures** — every number in
an answer must appear verbatim (after normalization) in the retrieved sources.
"""

import argparse
import asyncio
import json
import logging
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from anthropic import AsyncAnthropic

from secrag.config import get_settings
from secrag.db import session_factory
from secrag.embedding import get_embedder
from secrag.evals.golden import load_golden, resolve_refs
from secrag.evals.metrics import percentile
from secrag.evals.run import CHANGELOG, RESULTS_DIR, _git_rev
from secrag.generation import (
    REFUSAL_SENTENCE,
    estimate_cost_usd,
    generate_answer,
)
from secrag.rerank import get_reranker
from secrag.retrieval.search import hybrid_search, rerank_results

log = logging.getLogger(__name__)

CHECKPOINT = Path("evals/.gen-partial.jsonl")
CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
PAREN_RE = re.compile(r"\([^)]*\)")

JUDGE_SYSTEM = (
    "You grade whether an answer is faithful to its sources. Faithful means every"
    " factual claim is supported by the numbered sources; citing outside knowledge"
    " or contradicting the sources is unfaithful. Reply with strict JSON only:"
    ' {"faithful": true or false, "reason": "<one short sentence>"}'
)


def extract_numbers(text: str) -> set[str]:
    """Normalized numeric tokens ('40.4', '56950'); citation markers stripped."""
    text = CITE_RE.sub(" ", text)
    return {m.group().replace(",", "") for m in NUM_RE.finditer(text)}


def fabricated_numbers(answer: str, source_contents: list[str]) -> set[str]:
    source_nums: set[str] = set()
    for content in source_contents:
        source_nums |= extract_numbers(content)
    return extract_numbers(answer) - source_nums


def key_figure_present(expected_answer: str, answer: str) -> bool | None:
    """Is at least one expected figure in the answer? None if expected has none."""
    expected_nums = extract_numbers(PAREN_RE.sub(" ", expected_answer))
    if not expected_nums:
        return None
    return bool(expected_nums & extract_numbers(answer))


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], bool]:
    """Cited source indices (1-based) and whether all are in range."""
    cited = [int(m) for m in CITE_RE.findall(answer)]
    return cited, all(1 <= c <= n_sources for c in cited) if cited else False


def is_refusal(answer: str) -> bool:
    return REFUSAL_SENTENCE.rstrip(".").lower() in answer.lower()


async def judge_faithfulness(
    client: AsyncAnthropic, model: str, question: str, sources: list[str], answer: str
) -> dict:
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
    msg = await client.messages.create(
        model=model,
        max_tokens=150,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n{numbered}\n\nQuestion: {question}\n\nAnswer: {answer}",
            }
        ],
    )
    cost = estimate_cost_usd(model, msg.usage.input_tokens, msg.usage.output_tokens) or 0.0
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        verdict = json.loads(text[text.index("{") : text.rindex("}") + 1])
        return {
            "faithful": bool(verdict["faithful"]),
            "reason": verdict.get("reason", ""),
            "cost_usd": cost,
        }
    except (ValueError, KeyError):
        return {"faithful": None, "reason": f"unparseable: {text[:100]}", "cost_usd": cost}


def _load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    rows = [
        json.loads(line)
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {r["id"]: r for r in rows}


async def run_generation_eval(label: str, rerank: bool = True) -> dict:
    settings = get_settings()
    questions = load_golden()
    answerable = [q for q in questions if q.answerable]
    embedder = get_embedder()
    reranker = get_reranker() if rerank else None
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    factory = session_factory()

    async with factory() as session:
        relevant_by_q = await resolve_refs(session, answerable)

    done = _load_checkpoint()
    if done:
        log.info("resuming: %d question(s) already answered", len(done))

    with CHECKPOINT.open("a", encoding="utf-8") as ckpt:
        for q in questions:
            if q.id in done:
                continue
            async with factory() as session:
                results = await hybrid_search(
                    session,
                    q.question,
                    embedder.embed_query(q.question),
                    k=settings.rerank_candidates if rerank else 5,
                )
            if rerank:
                results = rerank_results(reranker, q.question, results, k=5)

            t0 = time.perf_counter()
            answer_parts: list[str] = []
            usage: dict = {}
            async with asyncio.timeout(120):
                async for ev in generate_answer(q.question, results):
                    if ev["type"] == "token":
                        answer_parts.append(ev["text"])
                    else:
                        usage = ev
            gen_ms = (time.perf_counter() - t0) * 1000
            answer = "".join(answer_parts)

            contents = [r.content for r in results]
            cited, cites_in_range = parse_citations(answer, len(results))
            relevant = relevant_by_q.get(q.id, set())
            cited_chunk_ids = {results[c - 1].chunk_id for c in cited if 1 <= c <= len(results)}
            row = {
                "id": q.id,
                "category": q.category,
                "answer": answer,
                "refused": is_refusal(answer),
                "cited": cited,
                "cites_in_range": cites_in_range,
                "cites_relevant": bool(cited_chunk_ids & relevant),
                "fabricated_numbers": sorted(fabricated_numbers(answer, contents)),
                "key_figure_present": key_figure_present(q.expected_answer, answer),
                "gen_ms": round(gen_ms, 1),
                "gen_cost_usd": usage.get("cost_usd") or 0.0,
            }
            if q.answerable and not row["refused"]:
                row["judge"] = await judge_faithfulness(
                    client, settings.generation_model, q.question, contents, answer
                )
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            done[q.id] = row
            log.info(
                "%s: refused=%s fabricated=%s judge=%s",
                q.id,
                row["refused"],
                row["fabricated_numbers"],
                row.get("judge", {}).get("faithful"),
            )

    rows = list(done.values())
    ans = [r for r in rows if r["category"] != "unanswerable"]
    unans = [r for r in rows if r["category"] == "unanswerable"]
    judged = [r for r in ans if r.get("judge", {}).get("faithful") is not None]
    numeric = [r for r in ans if r["key_figure_present"] is not None]
    total_cost = sum(r["gen_cost_usd"] + r.get("judge", {}).get("cost_usd", 0.0) for r in rows)

    result = {
        "label": label,
        "kind": "generation",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "config": {
            "mode": "generation",
            "generation_model": settings.generation_model,
            "retrieval": "hybrid" + ("+rerank" if rerank else ""),
            "k": 5,
        },
        "aggregate": {
            "n_answerable": len(ans),
            "n_unanswerable": len(unans),
            "faithful_rate": statistics.mean(r["judge"]["faithful"] for r in judged)
            if judged
            else None,
            "fabricated_figures_rate": statistics.mean(bool(r["fabricated_numbers"]) for r in ans),
            "key_figure_rate": statistics.mean(r["key_figure_present"] for r in numeric)
            if numeric
            else None,
            "citation_in_range_rate": statistics.mean(r["cites_in_range"] for r in ans),
            "citation_relevant_rate": statistics.mean(r["cites_relevant"] for r in ans),
            "wrongful_refusal_rate": statistics.mean(r["refused"] for r in ans),
            "refusal_correct_rate": statistics.mean(r["refused"] for r in unans) if unans else None,
        },
        "latency_ms": {
            "gen_p50": round(percentile([r["gen_ms"] for r in rows], 50), 1),
            "gen_p95": round(percentile([r["gen_ms"] for r in rows], 95), 1),
        },
        "total_cost_usd": round(total_cost, 4),
        "cost_per_query_usd": round(total_cost / len(rows), 6) if rows else 0.0,
        "per_question": rows,
    }
    return result


def write_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{result['created_at'][:10]}-{result['label']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    a = result["aggregate"]
    faithful = f"{a['faithful_rate']:.3f}" if a["faithful_rate"] is not None else "n/a"
    line = (
        f"- `{result['created_at'][:10]}` **{result['label']}** ({result['git_rev']},"
        f" {result['config']['generation_model']}): faithful {faithful}"
        f" · fabricated figures {a['fabricated_figures_rate']:.3f}"
        f" · refusal-correct {a['refusal_correct_rate']:.2f}"
        f" · ${result['total_cost_usd']:.4f} total\n"
    )
    CHANGELOG.write_text(CHANGELOG.read_text(encoding="utf-8") + line, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation evals (spends API credits)")
    parser.add_argument("--label", required=True)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = asyncio.run(run_generation_eval(args.label, rerank=not args.no_rerank))
    out = write_result(result)
    CHECKPOINT.unlink(missing_ok=True)
    log.info("wrote %s — total API spend $%.4f", out, result["total_cost_usd"])


if __name__ == "__main__":
    main()
