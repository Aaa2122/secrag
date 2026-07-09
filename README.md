# secrag — Production RAG over SEC 10-K filings

Ask questions to the annual reports of Apple, NVIDIA, Tesla & co — with the
engineering that makes it trustworthy: hybrid retrieval (pgvector + full-text),
local cross-encoder reranking, streamed answers with mandatory citations, and a
**public evals page** (recall@k before/after reranking, latency, cost per query)
backing every architecture decision with measurements.

> Status: **retrieval core complete (Jalons 0–5)** — 19 filings ingested
> (8,001 chunks), hybrid search + reranking, evals harness with versioned runs.
> Next: generation with citations (Jalon 6). Full case study at Jalon 10.

## Architecture

```
EDGAR ──ingest──▶ Postgres 17 + pgvector ──▶ FastAPI
 (10-K HTML,        documents / chunks         /search  vector | hybrid (RRF) + filters
  item-aware        (vector 384, tsvector,     /ask     streaming + citations
  chunking)          JSONB metadata)           /evals   versioned eval runs (public)

embeddings: local bge-small-en-v1.5 · reranker: local bge-reranker-v2-m3
generation: Claude API · observability: Langfuse
```

## Quickstart

```bash
docker compose up -d db          # Postgres 17 + pgvector on :5433
uv sync
uv run alembic upgrade head
uv run pytest                    # unit + integration (integration self-skips if db is down)

# download + ingest filings (declared User-Agent, <=10 req/s per SEC fair-access policy)
uv run python -m secrag.ingestion.download NVDA AAPL MSFT --years 2
uv run python -m secrag.ingestion.pipeline data/raw

docker compose up -d --build api # API on :8000 (GET /health)
curl "localhost:8000/search?q=nvidia+data+center+revenue+growth&mode=hybrid&rerank=false&k=3"

# evals (the deliverable): recall@k, MRR, latency, cost — versioned in evals/results/
uv run python -m secrag.evals.run --mode hybrid --rerank --label my-run
```

## Roadmap

| Jalon | Scope | Status |
|---|---|---|
| 0 | Skeleton: compose, schema (vector + tsvector + JSONB), CI, migrations | ✅ |
| 1 | Ingestion: EDGAR download, item-aware parsing/chunking, local embeddings | ✅ |
| 2 | Vector retrieval baseline + golden dataset (58 quote-anchored Q/A) | ✅ |
| 3 | Evals harness: recall@k, MRR, latency p50/p95, cost/query — versioned runs | ✅ |
| 4 | Hybrid search: tsvector + RRF fusion + metadata filters | ✅ |
| 5 | Reranking: top-30 → bge-reranker-v2-m3 → top-5 (the headline eval) | ✅ |
| 6 | Generation: /ask streaming, mandatory citations, refusals, faithfulness evals | ◐ |
| 7 | Observability (Langfuse), rate limiting, timeouts, fallbacks | ○ |
| 8 | Minimal UI + public /evals page | ◐ (deploy at 9) |
| 9 | Deploy (TLS + domain), CI gate: evals block merge on regression | ○ |
| 10 | README as case study: decisions, failure modes, costs | ○ |

## Evals (retrieval)

58-question golden dataset over 10 companies × 2 fiscal years: factual-numeric,
qualitative, cross-document and unanswerable categories. References are
**verbatim quotes** resolved to chunk ids at eval time, so the dataset survives
re-chunking. Full runs in [evals/results/](evals/results/), one line per run in
[evals/CHANGELOG.md](evals/CHANGELOG.md).

| run | recall@1 | recall@5 | recall@10 | MRR@10 | total p95 | $/query |
|---|---|---|---|---|---|---|
| ① vector-baseline | 0.266 | 0.484 | 0.580 | 0.424 | 70 ms | $0 |
| ② hybrid-rrf | 0.324 | 0.535 | 0.638 | 0.473 | 69 ms | $0 |
| ③ hybrid-rerank-v2m3 | **0.474** | **0.702** | 0.747 | **0.622** | 57.3 s | $0 |

Reading, honestly:

- **Hybrid fusion is free quality** — +5.1 pts recall@5 over pure vector at
  identical latency. Exact terms ("162%", "$40.4 billion", product names) are
  where embeddings alone miss and `tsvector` shines.
- **The reranker is the headline** — +16.7 pts recall@5 and +14.9 pts MRR over
  hybrid. Per category: qualitative 0.841, factual-numeric 0.717.
- **Its cost is real**: p50 33.5 s / p95 57.2 s per query for 30 candidates on
  CPU-only hardware. Mitigations, in order: GPU (two orders of magnitude),
  a smaller cross-encoder, or reranking top-10 — each re-measurable here.
- **Cross-document questions stay hard** (recall@5 0.292): a single fused
  ranking rarely surfaces chunks from *two* filings in five slots. The fix is
  query decomposition / per-ticker sub-retrieval, not a better reranker —
  scoped as future work.

## Design notes

- **Why 10-Ks:** public domain, universally understood demo, and genuinely hard:
  100–300 page documents, cross-references, financial tables, standardized
  Items enabling metadata-filtered retrieval ("compare NVDA vs AMD AI risk").
- **Why local embeddings/reranker:** the retrieval stack costs $0 per query and
  runs in CI without secrets; only generation calls a paid API. Swappable via
  config (`Embedder` interface) — trade-offs are measured on the evals page.
- **Evals-first:** the golden dataset was built *before* hybrid search and
  reranking landed, so every retrieval change ships with a before/after number.
- **Chunking:** blocks are packed to ~400 tokens without ever crossing Item
  boundaries; tables are serialized to markdown as dedicated chunks; every
  chunk is prefixed with `[TICKER 10-K FYxxxx — Item N: Title]` which grounds
  embeddings, feeds exact terms to full-text search, and makes citations
  self-describing.

## Failure modes found on real filings (and handling)

- **Item detection traps:** the table of contents repeats every Item heading
  (it lives in a `<table>` → excluded), cross-references like *"Refer to Item
  1A"* sit inside long paragraphs (length rule), and some filers wrap headings
  in layout tables (single-column tables degrade to text so headings stay
  detectable). Item ids must be monotonically increasing; anything else is
  body text.
- **Incorporation by reference:** NVIDIA files its financial statements under
  Item 15, JPMorgan and Exxon place most narrative content after Items 15/16.
  Content is fully captured and searchable; only the `item` metadata filter is
  approximate for those filers — flagged by the per-filing parse report.
- **EDGAR pagination:** high-volume filers (JPM) have short `recent` windows;
  older filings live in additional submission pages (not yet fetched — corpus
  is 19/20 filings).

Spec and per-jalon plans live in [docs/superpowers/](docs/superpowers/).
