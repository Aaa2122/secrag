# Production RAG on SEC 10-K Filings — Design Spec

**Date:** 2026-07-07
**Status:** Approved direction (roadmap authored by Auguste); technical defaults chosen by Claude are marked `[default — swappable]` and await confirmation.

## 1. Goal

A production-grade documentary assistant over a real vertical corpus — SEC annual
reports (10-K) — deployed with a public URL. The **key deliverable is the public
`/evals` page**: recall@k before/after reranking, MRR, latency p50/p95, cost per
query, versioned run history with a changelog. The chat UI is scenery around it.

This is a portfolio/case-study project feeding an Upwork gig and a Malt profile
("Production RAG on financial documents"). Every engineering decision should be
tellable in the README.

**Language:** all code, docs, UI and README in English (international client
audience). `[default — swappable]`

## 2. Corpus

10-K filings from EDGAR, ~10 well-known companies (AAPL, MSFT, NVDA, TSLA, AMZN,
GOOGL, META, AMD, JPM, XOM) × 2 fiscal years ≈ 20 documents, 100–300 pages each.

- Public domain → zero copyright risk for a public URL.
- EDGAR access: `data.sec.gov` submissions API + `www.sec.gov/Archives` for
  documents. Declared `User-Agent` header (name + email), ≤10 req/s rate limit,
  polite retry/backoff. No scraping hacks.
- Standardized structure (Item 1 Business, 1A Risk Factors, 7 MD&A, 8 Financial
  Statements…) → chunking by Item/section instead of blind token windows.
- Financial tables → serialized to markdown/structured text at ingestion.

## 3. Architecture

```
                        ┌─────────────────────────────────────────┐
 EDGAR ──ingest CLI──▶  │ Postgres 17 + pgvector                  │
 (httpx, rate-limited)  │  documents / chunks(embedding, tsv,     │
                        │  metadata JSONB) / eval tables          │
                        └───────────────┬─────────────────────────┘
                                        │
                   ┌────────────────────┴───────────────────┐
                   │ FastAPI (async, SQLAlchemy 2 + asyncpg)│
                   │  /search  vector | hybrid(RRF) + filters│
                   │  /ask     streaming + citations         │
                   │  /evals   public results page           │
                   └────────────────────┬───────────────────┘
                                        │
              local embeddings (bge)  ──┤──  local reranker (bge-reranker-v2-m3)
              Claude API (generation) ──┘    Langfuse (tracing, Jalon 7)
```

## 4. Technical decisions

| Area | Decision | Rationale |
|---|---|---|
| Package mgmt | `uv`, Python 3.12, src-layout package `secrag` | Fast, modern, reproducible; good signal |
| API | FastAPI + SQLAlchemy 2 async + asyncpg + Alembic | Standard production stack |
| DB | `pgvector/pgvector:pg17` via docker-compose | As specified |
| Embeddings | **Local** `BAAI/bge-small-en-v1.5` (384-dim) behind an `Embedder` interface; OpenAI `text-embedding-3-small` adapter as config switch `[default — swappable]` | Zero API key needed for Jalons 1–5, $0/query retrieval stack (a differentiator on the evals page), CI runs without secrets. Corpus+queries are English. A modest embedder also makes the reranker's lift more visible in evals. Swap = config change + re-ingest (~20 docs, minutes). |
| Vector index | HNSW, cosine distance | pgvector default best practice |
| Full-text | `tsvector` generated column + GIN, `english` config; `ts_rank` | Jalon 4 hybrid |
| Fusion | Reciprocal Rank Fusion (k=60) vector ∪ full-text, metadata filters (ticker, fiscal_year, item) as SQL WHERE | As specified |
| Reranker | `BAAI/bge-reranker-v2-m3` local, top-30 → top-5 | As specified (user choice) |
| Generation | Claude API, `claude-opus-4-8` default, env-configurable; streaming; mandatory citations; refusal when chunks lack the answer. Native citations feature (`citations: {enabled: true}` on document blocks) evaluated vs. prompt-based `[n]` markers at Jalon 6. | Env default per current API guidance; cost/quality sweep across models is itself evals-page content. |
| Eval judging | LLM-as-judge via **Batches API** (50% cost) for faithfulness; strict string/number comparison for cited figures (0% numeric-error target) | Financial data: a hallucinated number is disqualifying |
| Observability | Langfuse **Cloud free tier** at Jalon 7 `[default — swappable]`; compose stays lean (self-hosted Langfuse v3 = 5+ extra services) | Keep infra proportionate |
| Rate limiting | Redis (added to compose at Jalon 7) | As specified |
| CI | GitHub Actions: ruff + pytest (unit); integration tests need Postgres service container; evals gate merge on regression (Jalon 9) | As specified |
| Deploy | VPS or Fly.io, compose as-is, TLS + domain (Jalon 9) | Decide at Jalon 9 |

## 5. Data model

```sql
documents (
  id            bigint PK,
  ticker        text NOT NULL,            -- 'NVDA'
  company_name  text NOT NULL,
  cik           text NOT NULL,            -- zero-padded 10
  filing_type   text NOT NULL DEFAULT '10-K',
  fiscal_year   int  NOT NULL,            -- FY the report covers
  accession_number text NOT NULL UNIQUE,  -- idempotency key
  source_url    text NOT NULL,
  filed_at      date,
  ingested_at   timestamptz,
  UNIQUE (ticker, filing_type, fiscal_year)
)

chunks (
  id           bigint PK,
  document_id  bigint FK -> documents ON DELETE CASCADE,
  chunk_index  int NOT NULL,             -- order within document
  content      text NOT NULL,
  token_count  int,
  embedding    vector(384),              -- dims follow embedder config
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  metadata     jsonb NOT NULL DEFAULT '{}',  -- {item, item_title, section_path, is_table}
  UNIQUE (document_id, chunk_index)
)
-- HNSW index on embedding (cosine), GIN on tsv, GIN on metadata,
-- btree on (document_id)
```

Eval artifacts (golden dataset, run results) live in the repo as versioned
JSON/JSONL (`evals/`), not in the DB — reviewable in PRs, diffable, and the
changelog is git history. Run results also get committed (`evals/results/`).

## 6. Chunking strategy (Jalon 1)

1. Parse filing HTML with selectolax/BeautifulSoup; locate Item boundaries via
   the standardized headings (regex over normalized text, tolerant of the many
   formatting variants across filers).
2. Within an Item, split on sub-headings; then pack paragraphs into chunks of
   ~250–600 tokens with small overlap, never crossing Item boundaries.
3. Tables: extract as markdown, kept as dedicated chunks flagged
   `is_table: true`, prefixed with their nearest caption/heading for context.
4. Every chunk carries `{ticker, fiscal_year, item, item_title}` metadata.

## 7. Retrieval & generation

- `/search`: mode `vector | hybrid`; optional filters ticker/fiscal_year/item;
  `top_k`. Hybrid = RRF(vector top-50, fts top-50) → top-k. Optional rerank
  stage: top-30 → cross-encoder → top-5.
- `/ask`: retrieve (hybrid+rerank) → prompt with numbered chunks → stream
  tokens (SSE) → citations required, resolvable to `[NVDA 10-K FY2025, Item 1A]`
  with expandable source text; explicit refusal sentence when retrieval doesn't
  support an answer.

## 8. Evals methodology (Jalons 2–3, re-run every jalon)

- **Golden dataset:** 60–100 questions → reference chunk ids + reference answer,
  mixing factual-numeric, qualitative, and cross-document comparative questions.
  Semi-auto generation (LLM proposes Q/A from sampled chunks), 100% manually
  validated. Stored `evals/golden.jsonl`, versioned.
- **Retrieval metrics:** recall@k (k=1,3,5,10), MRR@10, per question category.
- **Ops metrics:** latency p50/p95 per stage (embed, search, rerank), cost/query.
- **Generation metrics (Jalon 6):** faithfulness (LLM-as-judge, batched),
  citation precision, numeric exactness (strict compare; target 0% error),
  refusal correctness on unanswerable questions (adversarial subset).
- **Harness:** `secrag evals run` CLI → JSON result versioned in
  `evals/results/YYYY-MM-DD-<label>.json` + row on `/evals` page. First run =
  vector baseline = first changelog line.

## 9. Milestones

Jalon 0 squelette (repo, compose, CI, migrations) → 1 ingestion EDGAR → 2
retrieval baseline + golden dataset → 3 evals harness → 4 hybrid+filters → 5
reranking → 6 generation+citations → 7 observability+hardening → 8 UI + /evals
page → 9 deploy+CI gate → 10 README/case study. Each jalon ends with an evals
re-run (from Jalon 3 onward) and a commit.

## 10. Risks / notes

- **10-K HTML variance** is the main technical risk (Item detection). Mitigation:
  tolerant regex + per-filing parse report (items found, sizes) + fallback to
  token-window chunking for unparseable sections, logged loudly.
- **Model downloads** (~130MB embedder, ~2.2GB reranker) cached in a docker
  volume / HF cache; CI mocks embeddings for unit tests, integration job pulls
  the small embedder only.
- **Windows dev machine**: everything runs via Docker; local venv for tests.
- Anthropic API key needed from Jalon 6 (generation + judge) and for golden
  dataset semi-auto generation (Jalon 2) — flag to Auguste before those jalons.

## Decisions awaiting Auguste's confirmation

1. Local bge embeddings as default (vs OpenAI/Voyage from day one).
2. English everywhere.
3. Langfuse Cloud (vs self-hosted) at Jalon 7.
4. Generation model default `claude-opus-4-8` (env-swappable; cost sweep later).
