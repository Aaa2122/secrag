from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from secrag.config import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        # pre_ping + recycle: idle connections through Docker's port proxy get
        # silently dropped; detect and replace them instead of failing mid-request.
        _engine = create_async_engine(
            get_settings().database_url, pool_pre_ping=True, pool_recycle=300
        )
    return _engine


def session_factory() -> async_sessionmaker:
    return async_sessionmaker(get_engine(), expire_on_commit=False)
