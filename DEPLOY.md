# Deploying secrag

The compose file ships as-is to any Linux VPS. Sizing: **4 GB RAM minimum**
for Postgres, Redis, the local embedder and the lightweight MiniLM reranker;
**8 GB recommended** for ingestion headroom.

## 1. VPS (Debian/Ubuntu) + Caddy TLS

```bash
# once: install docker + compose plugin, then
git clone https://github.com/Aaa2122/secrag && cd secrag

cat > .env <<EOF
SEC_USER_AGENT=secrag/0.1 (you@example.com)
ANTHROPIC_API_KEY=sk-ant-...
GENERATION_MODEL=claude-haiku-4-5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
EOF

docker compose up -d --build        # db + redis + api on :8000

# ingest the corpus inside the api container (models download on first run)
docker compose exec api uv run python -m secrag.ingestion.download \
  AAPL MSFT NVDA TSLA AMZN GOOGL META AMD JPM XOM --years 2
docker compose exec api uv run alembic upgrade head
docker compose exec api uv run python -m secrag.ingestion.pipeline data/raw
```

TLS + domain with Caddy (auto-provisions Let's Encrypt):

```bash
apt install caddy
cat > /etc/caddy/Caddyfile <<EOF
secrag.example.com {
    reverse_proxy localhost:8000
}
EOF
systemctl reload caddy
```

Point an A record at the VPS; done — `https://secrag.example.com/evals` is public.

## 2. Operational notes

- **Reranking runs locally on CPU**. MiniLM-L6 measured p50 3.7 s / p95 6.2 s
  for 30 candidates, versus p95 140.3 s for the previous BGE v2-m3 model.
  The evals page reports both configurations.
- **Rate limits** (per client IP, Redis): 30/min on `/search`, 10/min on
  `/ask`. Fail-open if Redis is down — the demo survives its limiter.
- **Generation budget**: `/ask` logs tokens + cost per request; with
  claude-haiku-4-5 a typical answer costs ~$0.0015.
- **CI gate**: every push re-checks committed eval artifacts against
  [evals/thresholds.json](evals/thresholds.json); merges are blocked on
  regression (`uv run python -m secrag.evals.gate`).
- Langfuse tracing is scoped for a follow-up (needs an account; free tier is
  enough): trace retrieval → rerank → generation per request.
