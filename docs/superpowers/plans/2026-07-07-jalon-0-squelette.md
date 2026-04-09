# Jalon 0 (Squelette) + EDGAR Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootable production skeleton — uv project, docker-compose (Postgres 17 + pgvector), FastAPI `/health`, Alembic migration creating `documents`/`chunks` with vector+tsvector+JSONB, CI workflow — plus a tested, rate-limited EDGAR client and download CLI (the key-free start of Jalon 1).

**Architecture:** src-layout package `secrag`; async SQLAlchemy 2 for the API path; sync httpx for the ingestion CLI; migrations own the schema (raw DDL where SQLAlchemy lacks expressiveness: HNSW index, generated tsvector). Integration tests hit the compose database and self-skip when it is unreachable, so `pytest` always works.

**Tech Stack:** Python 3.12, uv, FastAPI, SQLAlchemy 2 (asyncpg), Alembic, pgvector, pydantic-settings, httpx + tenacity + respx, ruff, pytest(-asyncio), Docker Compose, GitHub Actions.

## Global Constraints

- Package name `secrag`, src-layout (`src/secrag/`); Python `>=3.12`.
- Embedding dimension **384** (`bge-small-en-v1.5`), single source in `secrag.config`.
- DB: `pgvector/pgvector:pg17`, host port **5433**; app connects via `postgresql+asyncpg://`.
- SEC: declared User-Agent `secrag/0.1 (auguste.sagaert@gmail.com)`, ≤10 req/s, retry with backoff on 429/5xx.
- All code/comments/docs in English. Conventional commit messages.
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Project scaffold (pyproject, ruff, pytest, package skeleton)

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/secrag/__init__.py`, `tests/__init__.py`, `tests/test_sanity.py`

**Interfaces:**
- Produces: importable package `secrag` with `__version__ = "0.1.0"`; `uv run pytest` and `uv run ruff check .` green.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "secrag"
version = "0.1.0"
description = "Production RAG over SEC 10-K filings — pgvector, hybrid search, reranking, evals"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sqlalchemy[asyncio]>=2.0.36",
    "asyncpg>=0.30",
    "alembic>=1.14",
    "pgvector>=0.3.6",
    "pydantic-settings>=2.6",
    "httpx>=0.28",
    "tenacity>=9.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "respx>=0.22",
    "ruff>=0.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/secrag"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-m 'not live'"
markers = [
    "integration: needs the compose Postgres running",
    "live: performs real network calls to SEC EDGAR",
]
```

`.python-version` → `3.12`. `.gitignore` → standard Python (venv, __pycache__, .env, data/, .pytest_cache, .ruff_cache).

- [ ] **Step 2: Package + sanity test**

`src/secrag/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/test_sanity.py`:
```python
import secrag


def test_package_importable():
    assert secrag.__version__ == "0.1.0"
```

- [ ] **Step 3: Install and run**

Run: `uv python install 3.12 && uv sync && uv run pytest -q && uv run ruff check .`
Expected: 1 passed; ruff clean.

- [ ] **Step 4: Commit** — `chore: bootstrap uv project skeleton`

---

### Task 2: Settings + FastAPI app with /health

**Files:**
- Create: `src/secrag/config.py`, `src/secrag/api/__init__.py`, `src/secrag/api/main.py`, `tests/test_health.py`, `.env.example`

**Interfaces:**
- Produces: `secrag.config.Settings` (fields `database_url: str`, `embedding_dim: int = 384`, `sec_user_agent: str`) and `get_settings()` (cached); `secrag.api.main.app` (FastAPI) with `GET /health` → `{"status": "ok", "version": <pkg version>}`.

- [ ] **Step 1: Write failing test**

`tests/test_health.py`:
```python
import httpx

from secrag.api.main import app


async def test_health_returns_ok():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]
```

Run: `uv run pytest tests/test_health.py -q` → FAIL (module not found).

- [ ] **Step 2: Implement**

`src/secrag/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://secrag:secrag@localhost:5433/secrag"
    embedding_dim: int = 384
    sec_user_agent: str = "secrag/0.1 (auguste.sagaert@gmail.com)"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`src/secrag/api/main.py`:
```python
from fastapi import FastAPI

from secrag import __version__

app = FastAPI(title="secrag", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
```

`.env.example`:
```
DATABASE_URL=postgresql+asyncpg://secrag:secrag@localhost:5433/secrag
SEC_USER_AGENT=secrag/0.1 (auguste.sagaert@gmail.com)
```

- [ ] **Step 3: Run** — `uv run pytest -q && uv run ruff check .` → all pass.
- [ ] **Step 4: Commit** — `feat: settings and FastAPI health endpoint`

---

### Task 3: docker-compose (pgvector) + Dockerfile

**Files:**
- Create: `docker-compose.yml`, `Dockerfile`, `.dockerignore`

**Interfaces:**
- Produces: `docker compose up -d db` → Postgres 17 + pgvector on `localhost:5433`, creds `secrag/secrag`, db `secrag`; `api` service builds and serves uvicorn on `:8000`.

- [ ] **Step 1: Write compose file**

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: secrag
      POSTGRES_PASSWORD: secrag
      POSTGRES_DB: secrag
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U secrag -d secrag"]
      interval: 5s
      timeout: 3s
      retries: 10

  api:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://secrag:secrag@db:5432/secrag
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_NO_DEV=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen
CMD ["uv", "run", "--no-sync", "uvicorn", "secrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`: `.git`, `.venv`, `data/`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.env`.

- [ ] **Step 3: Verify**

Run: `docker compose up -d db && docker compose ps` → db healthy.
Run: `docker compose up -d --build api && curl -s localhost:8000/health` → `{"status":"ok",...}`.

- [ ] **Step 4: Commit** — `feat: docker-compose with pgvector and API image`

---

### Task 4: ORM models + Alembic initial migration

**Files:**
- Create: `src/secrag/db.py`, `src/secrag/models.py`, `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_initial_schema.py`, `tests/test_migrations.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `get_settings().database_url`.
- Produces: ORM classes `secrag.models.Document` / `Chunk` (attribute `meta` ↔ column `metadata`); `secrag.db.get_engine()`, `secrag.db.session_factory()`; command `uv run alembic upgrade head`.

- [ ] **Step 1: Write models**

`src/secrag/db.py`:
```python
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from secrag.config import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_settings().database_url)
    return _engine


def session_factory() -> async_sessionmaker:
    return async_sessionmaker(get_engine(), expire_on_commit=False)
```

`src/secrag/models.py`:
```python
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, Date, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("ticker", "filing_type", "fiscal_year"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text)
    company_name: Mapped[str] = mapped_column(Text)
    cik: Mapped[str] = mapped_column(Text)
    filing_type: Mapped[str] = mapped_column(Text, default="10-K")
    fiscal_year: Mapped[int] = mapped_column(Integer)
    accession_number: Mapped[str] = mapped_column(Text, unique=True)
    source_url: Mapped[str] = mapped_column(Text)
    filed_at: Mapped[date | None] = mapped_column(Date)
    ingested_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")

    document: Mapped[Document] = relationship(back_populates="chunks")
```

- [ ] **Step 2: Alembic setup (async env)**

`alembic.ini` (minimal): `[alembic]` `script_location = migrations`, logging defaults.

`migrations/env.py`:
```python
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from secrag.config import get_settings
from secrag.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"),
                      target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(config.get_section(config.config_ini_section, {}),
                                      prefix="sqlalchemy.")
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
```

`migrations/versions/0001_initial_schema.py`: `CREATE EXTENSION IF NOT EXISTS vector`, create both tables (mirroring the models exactly), then raw DDL indexes:

```python
op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
op.execute("CREATE INDEX ix_chunks_metadata ON chunks USING gin (metadata)")
```

- [ ] **Step 3: Write integration test (self-skipping)**

`tests/conftest.py`:
```python
import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from secrag.config import get_settings


def _db_reachable() -> bool:
    async def probe() -> bool:
        try:
            engine = create_async_engine(get_settings().database_url)
            async with engine.connect():
                pass
            await engine.dispose()
            return True
        except Exception:
            return False

    return asyncio.run(probe())


def pytest_collection_modifyitems(config, items):
    if _db_reachable():
        return
    skip = pytest.mark.skip(reason="database not reachable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
```

`tests/test_migrations.py`:
```python
import pytest
from sqlalchemy import text

from secrag.db import get_engine

pytestmark = pytest.mark.integration


async def test_schema_tables_and_indexes_exist():
    engine = get_engine()
    async with engine.connect() as conn:
        tables = (await conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        ))).scalars().all()
        assert {"documents", "chunks"} <= set(tables)

        indexes = (await conn.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename='chunks'"
        ))).scalars().all()
        assert "ix_chunks_embedding_hnsw" in indexes
        assert "ix_chunks_tsv" in indexes


async def test_chunk_roundtrip_with_vector_and_tsv():
    from secrag.db import session_factory
    from secrag.models import Chunk, Document

    async with session_factory()() as session:
        doc = Document(ticker="TEST", company_name="Test Co", cik="0000000000",
                       fiscal_year=2099, accession_number="test-acc-1",
                       source_url="http://example.com")
        session.add(doc)
        await session.flush()
        session.add(Chunk(document_id=doc.id, chunk_index=0,
                          content="Revenue grew twelve percent",
                          embedding=[0.1] * 384, meta={"item": "7"}))
        await session.commit()

    async with session_factory()() as session:
        row = (await session.execute(text(
            "SELECT tsv IS NOT NULL, embedding IS NOT NULL FROM chunks "
            "WHERE content = 'Revenue grew twelve percent'"
        ))).one()
        assert row == (True, True)
        await session.execute(text("DELETE FROM documents WHERE ticker = 'TEST'"))
        await session.commit()
```

- [ ] **Step 4: Apply and verify**

Run: `docker compose up -d db && uv run alembic upgrade head && uv run pytest -q`
Expected: migration applies; all tests pass (integration included).

- [ ] **Step 5: Commit** — `feat: documents/chunks schema with pgvector, tsvector, JSONB via alembic`

---

### Task 5: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: on push/PR — ruff lint+format check, full pytest against a pgvector service container with migrations applied.

- [ ] **Step 1: Write workflow**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg17
        env:
          POSTGRES_USER: secrag
          POSTGRES_PASSWORD: secrag
          POSTGRES_DB: secrag
        ports: ["5433:5432"]
        options: >-
          --health-cmd "pg_isready -U secrag -d secrag"
          --health-interval 5s --health-timeout 3s --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run alembic upgrade head
      - run: uv run pytest -q
```

- [ ] **Step 2: Validate YAML locally** — `uv run python -c "import yaml,pathlib;yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())"` (PyYAML ships transitively; if absent, visual check suffices).
- [ ] **Step 3: Commit** — `ci: ruff + pytest with pgvector service`

---

### Task 6: EDGAR client (ticker→CIK, 10-K discovery, rate-limited download)

**Files:**
- Create: `src/secrag/ingestion/__init__.py`, `src/secrag/ingestion/edgar.py`, `tests/test_edgar.py`

**Interfaces:**
- Produces:
  - `Filing` dataclass: `accession_number: str`, `filing_date: str`, `report_date: str`, `primary_document: str`, `form: str`, plus properties `fiscal_year: int` (year of `report_date`) and `accession_nodash: str`.
  - `parse_10k_filings(submissions: dict) -> list[Filing]` (pure; newest first; forms `10-K` only, excludes `10-K/A`).
  - `EdgarClient(user_agent: str | None = None)` with `.get_cik(ticker) -> str` (zero-padded 10), `.get_submissions(cik) -> dict`, `.download_filing_html(cik, filing) -> str`, throttled to ≤10 req/s, tenacity retry (5 attempts, exponential) on 429/5xx.
  - `primary_doc_url(cik: str, filing: Filing) -> str`.

- [ ] **Step 1: Write failing tests**

`tests/test_edgar.py`:
```python
import httpx
import pytest
import respx

from secrag.ingestion.edgar import EdgarClient, Filing, parse_10k_filings, primary_doc_url

SUBMISSIONS = {
    "cik": 1045810,
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001045810-25-000023", "0001045810-24-000029", "0001045810-24-000030"],
            "form": ["10-K", "10-K/A", "10-K"],
            "filingDate": ["2025-02-26", "2024-05-01", "2024-02-21"],
            "reportDate": ["2025-01-26", "2024-01-28", "2024-01-28"],
            "primaryDocument": ["nvda-20250126.htm", "nvda-20240128a.htm", "nvda-20240128.htm"],
        }
    },
}


def test_parse_10k_filings_filters_and_orders():
    filings = parse_10k_filings(SUBMISSIONS)
    assert [f.form for f in filings] == ["10-K", "10-K"]
    assert filings[0].accession_number == "0001045810-25-000023"
    assert filings[0].fiscal_year == 2025
    assert filings[1].fiscal_year == 2024


def test_primary_doc_url():
    filing = Filing(accession_number="0001045810-25-000023", filing_date="2025-02-26",
                    report_date="2025-01-26", primary_document="nvda-20250126.htm", form="10-K")
    assert primary_doc_url("0001045810", filing) == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )


@respx.mock
def test_get_cik_pads_to_ten_digits():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json={
            "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        })
    )
    client = EdgarClient(user_agent="test-agent test@example.com")
    assert client.get_cik("nvda") == "0001045810"
    with pytest.raises(KeyError):
        client.get_cik("NOPE")


@respx.mock
def test_download_retries_on_429_then_succeeds():
    route = respx.get(
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    ).mock(side_effect=[httpx.Response(429), httpx.Response(200, text="<html>10-K</html>")])
    client = EdgarClient(user_agent="test-agent test@example.com")
    filing = Filing(accession_number="0001045810-25-000023", filing_date="2025-02-26",
                    report_date="2025-01-26", primary_document="nvda-20250126.htm", form="10-K")
    html = client.download_filing_html("0001045810", filing)
    assert "10-K" in html
    assert route.call_count == 2
```

Run: `uv run pytest tests/test_edgar.py -q` → FAIL (module missing).

- [ ] **Step 2: Implement `edgar.py`**

```python
"""SEC EDGAR access: fair-access compliant (declared UA, <=10 req/s, backoff)."""

import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from secrag.config import get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
MIN_INTERVAL_S = 0.11  # ~9 req/s, under SEC's 10 req/s fair-access limit


@dataclass(frozen=True)
class Filing:
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str
    form: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")


def parse_10k_filings(submissions: dict) -> list[Filing]:
    recent = submissions["filings"]["recent"]
    filings = [
        Filing(
            accession_number=recent["accessionNumber"][i],
            filing_date=recent["filingDate"][i],
            report_date=recent["reportDate"][i],
            primary_document=recent["primaryDocument"][i],
            form=recent["form"][i],
        )
        for i in range(len(recent["form"]))
        if recent["form"][i] == "10-K"
    ]
    return sorted(filings, key=lambda f: f.filing_date, reverse=True)


def primary_doc_url(cik: str, filing: Filing) -> str:
    return ARCHIVES_URL.format(cik_int=int(cik), acc_nodash=filing.accession_nodash,
                               doc=filing.primary_document)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or get_settings().sec_user_agent
        self._http = httpx.Client(headers={"User-Agent": ua}, timeout=30.0,
                                  follow_redirects=True)
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, max=30), reraise=True)
    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp

    def get_cik(self, ticker: str) -> str:
        data = self._get(TICKERS_URL).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return f"{entry['cik_str']:010d}"
        raise KeyError(f"ticker not found on EDGAR: {ticker}")

    def get_submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def download_filing_html(self, cik: str, filing: Filing) -> str:
        return self._get(primary_doc_url(cik, filing)).text
```

- [ ] **Step 3: Run** — `uv run pytest tests/test_edgar.py -q` → 4 passed. Note: the 429-retry test sleeps ~1s (first backoff) — acceptable.
- [ ] **Step 4: Commit** — `feat: EDGAR client with 10-K discovery, throttling and retry`

---

### Task 7: Download CLI + live smoke test

**Files:**
- Create: `src/secrag/ingestion/download.py`, `tests/test_download_cli.py`

**Interfaces:**
- Consumes: `EdgarClient`, `parse_10k_filings`, `primary_doc_url`.
- Produces: `download_10ks(tickers: list[str], years: int, dest: Path, client: EdgarClient | None = None) -> list[Path]` — writes `data/raw/{TICKER}/FY{fiscal_year}_{accession_nodash}.html` plus a sidecar `.json` (ticker, cik, company_name, accession, dates, url); skips files that already exist (idempotent). CLI: `uv run python -m secrag.ingestion.download NVDA AAPL --years 2`.

- [ ] **Step 1: Write failing test (mocked client)**

`tests/test_download_cli.py`:
```python
import json

import httpx
import respx

from secrag.ingestion.download import download_10ks
from secrag.ingestion.edgar import EdgarClient

TICKERS_JSON = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
SUBMISSIONS = {
    "name": "NVIDIA CORP",
    "filings": {"recent": {
        "accessionNumber": ["0001045810-25-000023", "0001045810-24-000030"],
        "form": ["10-K", "10-K"],
        "filingDate": ["2025-02-26", "2024-02-21"],
        "reportDate": ["2025-01-26", "2024-01-28"],
        "primaryDocument": ["nvda-20250126.htm", "nvda-20240128.htm"],
    }},
}


@respx.mock
def test_download_10ks_writes_files_and_sidecars(tmp_path):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS_JSON))
    respx.get("https://data.sec.gov/submissions/CIK0001045810.json").mock(
        return_value=httpx.Response(200, json=SUBMISSIONS))
    respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
        return_value=httpx.Response(200, text="<html>fake 10-K</html>"))

    client = EdgarClient(user_agent="test test@example.com")
    written = download_10ks(["NVDA"], years=1, dest=tmp_path, client=client)

    assert len(written) == 1
    html_path = tmp_path / "NVDA" / "FY2025_000104581025000023.html"
    assert html_path.read_text(encoding="utf-8") == "<html>fake 10-K</html>"
    sidecar = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["ticker"] == "NVDA"
    assert sidecar["cik"] == "0001045810"
    assert sidecar["fiscal_year"] == 2025

    # idempotent: second run downloads nothing new
    assert download_10ks(["NVDA"], years=1, dest=tmp_path, client=client) == []
```

- [ ] **Step 2: Implement `download.py`**

```python
"""Download 10-K primary documents from EDGAR into data/raw/."""

import argparse
import json
import logging
from pathlib import Path

from secrag.ingestion.edgar import EdgarClient, parse_10k_filings, primary_doc_url

log = logging.getLogger(__name__)


def download_10ks(tickers: list[str], years: int, dest: Path,
                  client: EdgarClient | None = None) -> list[Path]:
    client = client or EdgarClient()
    written: list[Path] = []
    for ticker in (t.upper() for t in tickers):
        cik = client.get_cik(ticker)
        submissions = client.get_submissions(cik)
        company_name = submissions.get("name", ticker)
        for filing in parse_10k_filings(submissions)[:years]:
            out = dest / ticker / f"FY{filing.fiscal_year}_{filing.accession_nodash}.html"
            if out.exists():
                log.info("skip (exists): %s", out)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(client.download_filing_html(cik, filing), encoding="utf-8")
            out.with_suffix(".json").write_text(json.dumps({
                "ticker": ticker, "cik": cik, "company_name": company_name,
                "filing_type": "10-K", "fiscal_year": filing.fiscal_year,
                "accession_number": filing.accession_number,
                "filing_date": filing.filing_date, "report_date": filing.report_date,
                "source_url": primary_doc_url(cik, filing),
            }, indent=2), encoding="utf-8")
            written.append(out)
            log.info("downloaded: %s", out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    written = download_10ks(args.tickers, args.years, args.dest)
    log.info("done: %d new file(s)", len(written))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests** — `uv run pytest -q` → all pass.
- [ ] **Step 4: Live smoke (one ticker, real EDGAR)**

Run: `uv run python -m secrag.ingestion.download NVDA --years 1`
Expected: one HTML (several MB) + sidecar under `data/raw/NVDA/`. (`data/` is gitignored.)

- [ ] **Step 5: Commit** — `feat: idempotent 10-K download CLI`

---

### Task 8: README stub + wrap-up

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README stub** — project one-liner, architecture sketch, quickstart (`docker compose up -d db`, `uv sync`, `uv run alembic upgrade head`, `uv run pytest`, download CLI example), roadmap table (Jalons 0–10 with status), note that `/evals` is the key deliverable. Full case study lands at Jalon 10.
- [ ] **Step 2: Final check** — `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` all green; `git status` clean after commit.
- [ ] **Step 3: Commit** — `docs: README with quickstart and roadmap`

## Self-review notes

- Spec coverage: Jalon 0 fully (repo ✅ compose ✅ CI ✅ migrations ✅ schema ✅); Langfuse intentionally deferred to Jalon 7 per spec §4; Jalon 1 partially (download only — parsing/chunking/embedding are the next plan).
- Types consistent: `Filing` fields used identically in Tasks 6–7; `EMBEDDING_DIM = 384` matches spec §5.
- No placeholders remain; every code step has full code.
