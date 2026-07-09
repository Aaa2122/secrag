# Jalons 1–5: Ingestion → Retrieval Core → Evals Implementation Plan

> Lean plan (tasks, files, interfaces, verification). Executed inline in-session
> immediately after writing; full code lives in the implementation commits.
> Spec: `docs/superpowers/specs/2026-07-07-sec-10k-rag-design.md`.

**Goal:** Corpus ingested (10 tickers × 2 FY) with item-aware chunking and local
embeddings; `/search` in vector and hybrid modes with metadata filters; local
cross-encoder reranking; golden dataset + evals harness producing versioned
recall@k / MRR / latency results — the first three rows of the evals changelog.

## Facts established by probing the real NVDA FY2026 filing

- Real Item headings: short standalone `<div>` blocks whose text starts with
  `Item N[.:]` (bold), **not** inside a `<table>`.
- TOC entries: `Item N.` inside `<a>` within a `<table>` → excluded by the
  table rule.
- Cross-references («Refer to "Item 1A..."») sit inside long paragraphs →
  excluded by the short-block rule (<120 chars).
- Item order must be monotonic (1, 1A, 1B, 1C, 2, 3, 4, 5, 6, 7, 7A, 8, 9,
  9A, 9B, 9C, 10–16); out-of-order matches are dropped.

## Task 1 — Embedder (local bge-small-en-v1.5 behind interface)

Files: `src/secrag/embedding.py`, `tests/test_embedding.py`.
Interface: `Embedder` protocol — `dim: int`,
`embed_passages(texts: list[str]) -> list[list[float]]`,
`embed_query(text: str) -> list[float]` (bge query instruction prefix
`"Represent this sentence for searching relevant passages: "`).
`get_embedder()` factory reads `settings.embedding_provider` (`local` only for
now; `openai` slot documented). Normalized embeddings (cosine).
Tests: unit (protocol/factory with a `FakeEmbedder`), `integration`-marked test
loading the real model asserting dim=384, |v|≈1, and that query≠passage vector.
New markers: `heavy` (reranker) added to addopts exclusion with `live`.

## Task 2 — 10-K parser + table serialization

Files: `src/secrag/ingestion/parse10k.py`, `tests/test_parse10k.py`,
`tests/fixtures/mini_10k.html` (synthetic, committed).
Interfaces:
- `Block = tuple[str, str]` (`kind` in {"text","table"}, text/markdown).
- `Section` dataclass: `item: str | None`, `title: str`, `blocks: list[Block]`.
- `parse_10k(html: str) -> list[Section]` — selectolax walk; block-level
  elements in document order; tables → markdown (header row heuristic, skip
  layout/empty tables); heading detection per the probed rules above.
- `parse_report(sections) -> dict` — items found, chars per item (ingest logs
  it; thin filings surface here).
Tests: synthetic fixture with TOC table + cross-reference traps; real-file
smoke (`integration`, skipped if `data/raw/NVDA/...` absent) asserting core
items {1, 1A, 7, 7A, 8} found and 1A is the largest-ish.

## Task 3 — Chunker

Files: `src/secrag/ingestion/chunker.py`, `tests/test_chunker.py`.
Interfaces:
- `ChunkDraft` dataclass: `content: str`, `token_count: int`, `meta: dict`
  (`item`, `item_title`, `is_table`).
- `chunk_sections(sections, *, ticker, fiscal_year, target_tokens=400,
  max_tokens=500) -> list[ChunkDraft]`.
- Approx tokens = `len(text) // 4`. Chunks never cross Item boundaries; each
  chunk content is prefixed with a context line
  `[{ticker} 10-K FY{fy} — Item {item}: {title}]` (helps embedding + fts;
  documented). Tables = dedicated chunks (split by rows if huge, header
  repeated); tiny trailing chunks merged back.
Tests: boundary respect, table isolation, context prefix, size bounds.

## Task 4 — Ingestion pipeline + corpus load

Files: `src/secrag/ingestion/pipeline.py`, `tests/test_pipeline.py`.
Interface: `ingest_directory(raw_dir: Path, *, force: bool = False,
embedder: Embedder | None = None) -> IngestStats`; CLI
`python -m secrag.ingestion.pipeline data/raw [--force]`.
Per filing (html + sidecar json): skip if accession exists with chunks and not
force; parse → chunk → embed (batch 64) → upsert document, replace chunks; set
`ingested_at`; log parse report + stats (docs, chunks, embed seconds, $0.00).
Test: end-to-end on the synthetic fixture with `FakeEmbedder` against the real
DB (`integration`); idempotency asserted.
Then: run on full `data/raw` corpus; record stats; spot-check via SQL.

## Task 5 — /search endpoint (vector baseline, filters-ready)

Files: `src/secrag/retrieval/__init__.py`, `src/secrag/retrieval/search.py`,
`src/secrag/api/schemas.py`, modify `src/secrag/api/main.py`,
`tests/test_search_api.py`, `tests/test_retrieval.py`.
Interfaces:
- `SearchFilters` dataclass: `tickers`, `fiscal_years`, `items` (all optional
  lists).
- `vector_search(session, query_embedding, k, filters) -> list[RetrievedChunk]`
  (`RetrievedChunk`: chunk_id, document_id, ticker, fiscal_year, item,
  item_title, content, score`).
- `GET /search?q=...&mode=vector&k=5&tickers=NVDA&fiscal_years=2026&items=1A`
  → `{results: [...], timing_ms: {embed, search}}`. Embedder via FastAPI
  dependency (overridden with fake in API tests).
Tests: retrieval fn against seeded DB rows (integration); API test with fake
embedder + seeded rows; live sanity query after corpus ingest.

## Task 6 — Golden dataset (quote-anchored)

Files: `evals/golden.jsonl` (committed), `src/secrag/evals/golden.py`,
`tests/test_golden.py`.
Record: `{id, category: factual_numeric|qualitative|cross_doc|unanswerable,
question, expected_answer, refs: [{ticker, fiscal_year, quote}]}` — quotes are
verbatim substrings of chunk content (post context-prefix content OK to match
via `content LIKE %quote%` scoped to the (ticker, fy) document). Resolution:
`resolve_refs(session, golden) -> dict[qid, set[chunk_id]]`; unresolved refs
fail loudly. ~60 questions: ≥4 per ticker, all categories, authored in-session
by Claude from actual DB chunks, flagged for Auguste's manual validation.
Tests: schema validation of every line; resolution rate == 100% (integration).

## Task 7 — Evals harness + baseline run (Jalon 3)

Files: `src/secrag/evals/metrics.py`, `src/secrag/evals/run.py`,
`tests/test_metrics.py`, results in `evals/results/` (committed),
`evals/CHANGELOG.md`.
Interfaces:
- `recall_at_k(relevant: set, retrieved: list, k) -> float`,
  `mrr_at_k(relevant, retrieved, k=10) -> float` (pure, unit-tested).
- CLI `python -m secrag.evals.run --mode vector --label vector-baseline`
  → runs all answerable goldens through retrieval, measures per-stage wall
  times, writes `evals/results/<date>-<label>.json`: config (git rev, embedder,
  mode, k), aggregate + per-category recall@{1,3,5,10}, MRR@10, latency
  p50/p95, cost_per_query_usd (0.0 retrieval). Changelog line per run.
Baseline run committed = first changelog entry.

## Task 8 — Hybrid search + filters (Jalon 4)

Files: `src/secrag/retrieval/search.py` (add `fulltext_search`, `rrf_fuse`,
`hybrid_search`), API `mode=hybrid` default k unchanged.
`fulltext_search`: `websearch_to_tsquery('english', q)` + `ts_rank_cd`, top-50.
`rrf_fuse(lists, k_rrf=60)`: standard RRF over chunk id lists.
Tests: RRF pure-function unit tests; hybrid integration on seeded rows (exact
term beats vector-only). Evals re-run `--mode hybrid --label hybrid-rrf` →
commit results + changelog.

## Task 9 — Reranker (Jalon 5)

Files: `src/secrag/rerank.py`, `tests/test_rerank.py` (`heavy` marker),
retrieval pipeline `search_pipeline(query, mode, k, rerank: bool)` returning
top-5 after cross-encoder over hybrid top-30; API `rerank=true` param.
`Reranker` protocol + `BgeReranker` (`BAAI/bge-reranker-v2-m3`, CPU, batch 8)
+ `FakeReranker` for tests. Honest latency measured into eval results.
Evals re-run `--mode hybrid --rerank --label hybrid-rerank-v2m3` → commit.
README roadmap statuses updated (Jalons 1–5 ✅) + evals table snippet.

## Verification gates (every task)

`uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
green before each commit; evals runs committed with their JSON; no
Co-Authored-By trailers (Auguste's explicit request); real dates.
