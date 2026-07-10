"""CI gate: fail when the latest committed eval runs regress below thresholds.

Reads only committed artifacts (evals/results/*.json + evals/thresholds.json),
so it needs neither the database nor any model — cheap enough to run on every
push, strict enough to block a merge that ships a retrieval regression.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("evals/results")
THRESHOLDS = Path("evals/thresholds.json")


def load_runs(results_dir: Path = RESULTS_DIR) -> tuple[dict | None, dict | None]:
    """Latest reranked-retrieval run and latest generation run (either may be None)."""
    retrieval, generation = None, None
    runs = sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in results_dir.glob("*.json")),
        key=lambda r: r["created_at"],
    )
    for run in runs:
        if run.get("kind") == "generation":
            generation = run
        elif run.get("config", {}).get("reranker"):
            retrieval = run
    return retrieval, generation


def check(retrieval: dict | None, generation: dict | None, thresholds: dict) -> list[str]:
    failures: list[str] = []
    t_ret = thresholds.get("retrieval", {})
    if retrieval is None:
        if t_ret:
            failures.append("no reranked retrieval run found in evals/results/")
    else:
        agg = retrieval["aggregate"]
        for metric, floor in t_ret.items():
            if agg[metric] < floor:
                failures.append(
                    f"retrieval {metric}={agg[metric]:.3f} < floor {floor} ({retrieval['label']})"
                )

    t_gen = thresholds.get("generation", {})
    if generation is not None:
        agg = generation["aggregate"]
        for metric, bound in t_gen.items():
            if metric.endswith("_max"):
                value = agg[metric.removesuffix("_max")]
                if value > bound:
                    failures.append(
                        f"generation {metric.removesuffix('_max')}={value:.3f} > cap {bound}"
                        f" ({generation['label']})"
                    )
            elif agg[metric] is not None and agg[metric] < bound:
                failures.append(
                    f"generation {metric}={agg[metric]:.3f} < floor {bound} ({generation['label']})"
                )
    return failures


def main() -> None:
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    retrieval, generation = load_runs()
    failures = check(retrieval, generation, thresholds)
    if failures:
        print("EVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    checked = [r["label"] for r in (retrieval, generation) if r]
    print(f"eval gate OK ({', '.join(checked)})")


if __name__ == "__main__":
    main()
