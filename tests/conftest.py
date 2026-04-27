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
