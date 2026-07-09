import json

from secrag.evals.gate import check, load_runs

RET_OK = {
    "label": "r1", "created_at": "2026-07-09T01:00:00+00:00",
    "config": {"reranker": "BAAI/bge-reranker-v2-m3"},
    "aggregate": {"recall@5": 0.70, "mrr@10": 0.62},
}
RET_OLD_VECTOR = {
    "label": "v1", "created_at": "2026-07-09T00:00:00+00:00",
    "config": {"reranker": None},
    "aggregate": {"recall@5": 0.48, "mrr@10": 0.42},
}
GEN_OK = {
    "label": "g1", "kind": "generation", "created_at": "2026-07-09T02:00:00+00:00",
    "config": {},
    "aggregate": {
        "fabricated_figures_rate": 0.0,
        "citation_in_range_rate": 1.0,
        "refusal_correct_rate": 1.0,
    },
}
THRESH = {
    "retrieval": {"recall@5": 0.65, "mrr@10": 0.55},
    "generation": {
        "fabricated_figures_rate_max": 0.05,
        "citation_in_range_rate": 0.9,
        "refusal_correct_rate": 0.8,
    },
}


def _write_runs(tmp_path, runs):
    for i, r in enumerate(runs):
        (tmp_path / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")


def test_load_runs_picks_latest_reranked_and_generation(tmp_path):
    _write_runs(tmp_path, [RET_OLD_VECTOR, RET_OK, GEN_OK])
    retrieval, generation = load_runs(tmp_path)
    assert retrieval["label"] == "r1"  # vector-only run is not the gate target
    assert generation["label"] == "g1"


def test_gate_passes_when_above_thresholds():
    assert check(RET_OK, GEN_OK, THRESH) == []


def test_gate_fails_on_retrieval_regression():
    bad = {**RET_OK, "aggregate": {"recall@5": 0.60, "mrr@10": 0.62}}
    failures = check(bad, None, THRESH)
    assert len(failures) == 1 and "recall@5" in failures[0]


def test_gate_fails_on_fabricated_figures():
    bad = {**GEN_OK, "aggregate": {**GEN_OK["aggregate"], "fabricated_figures_rate": 0.10}}
    failures = check(RET_OK, bad, THRESH)
    assert len(failures) == 1 and "fabricated_figures_rate" in failures[0]


def test_gate_requires_a_retrieval_run():
    assert check(None, None, THRESH) == ["no reranked retrieval run found in evals/results/"]
