# secrag — Production RAG over SEC 10-K filings

Ask questions to the annual reports of Apple, NVIDIA, Tesla & co — with the
engineering that makes it trustworthy: hybrid retrieval (pgvector + full-text),
local cross-encoder reranking, streamed answers with mandatory citations, and a
**public evals page** (recall@k before/after reranking, latency, cost per query)
backing every architecture decision with measurements.

> Status: **Jalon 0 complete** — skeleton, schema, CI, EDGAR downloader.
> The full case study lands at Jalon 10.

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

# download filings (declared User-Agent, <=10 req/s per SEC fair-access policy)
uv run python -m secrag.ingestion.download NVDA AAPL MSFT --years 2

docker compose up -d --build api # API on :8000 (GET /health)
```

## Roadmap

| Jalon | Scope | Status |
|---|---|---|
| 0 | Skeleton: compose, schema (vector + tsvector + JSONB), CI, migrations | ✅ |
| 1 | Ingestion: EDGAR download ✅ · item-aware parsing/chunking · embeddings | ◐ |
| 2 | Vector retrieval baseline + golden dataset (60–100 validated Q/A) | ○ |
| 3 | Evals harness: recall@k, MRR, latency p50/p95, cost/query — versioned runs | ○ |
| 4 | Hybrid search: tsvector + RRF fusion + metadata filters | ○ |
| 5 | Reranking: top-30 → bge-reranker-v2-m3 → top-5 (the headline eval) | ○ |
| 6 | Generation: /ask streaming, mandatory citations, refusals, faithfulness evals | ○ |
| 7 | Observability (Langfuse), rate limiting, timeouts, fallbacks | ○ |
| 8 | Minimal UI + public /evals page | ○ |
| 9 | Deploy (TLS + domain), CI gate: evals block merge on regression | ○ |
| 10 | README as case study: decisions, failure modes, costs | ○ |

## Design notes

- **Why 10-Ks:** public domain, universally understood demo, and genuinely hard:
  100–300 page documents, cross-references, financial tables, standardized
  Items enabling metadata-filtered retrieval ("compare NVDA vs AMD AI risk").
- **Why local embeddings/reranker:** the retrieval stack costs $0 per query and
  runs in CI without secrets; only generation calls a paid API. Swappable via
  config (`Embedder` interface) — trade-offs are measured on the evals page.
- **Evals-first:** the golden dataset is built *before* hybrid search and
  reranking land, so every retrieval change ships with a before/after number.

Spec and per-jalon plans live in [docs/superpowers/](docs/superpowers/).
