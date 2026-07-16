FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_NO_DEV=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project && uv cache clean
COPY . .
RUN uv sync --frozen && uv cache clean
CMD ["uv", "run", "--no-sync", "uvicorn", "secrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
