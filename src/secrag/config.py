from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://secrag:secrag@localhost:5433/secrag"
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 30
    generation_model: str = "claude-opus-4-8"
    anthropic_api_key: str | None = None  # falls back to the SDK's env resolution
    sec_user_agent: str = "secrag/0.1 (auguste.sagaert@gmail.com)"


@lru_cache
def get_settings() -> Settings:
    return Settings()
