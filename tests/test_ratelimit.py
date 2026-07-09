from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from secrag.api import ratelimit


class FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, ttl: int) -> None:
        pass


class BrokenRedis:
    async def incr(self, key: str) -> int:
        raise ConnectionError("redis down")


def _request(ip: str = "1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=ip))


async def test_blocks_after_limit(monkeypatch):
    monkeypatch.setattr(ratelimit, "get_redis", lambda: FakeRedis())
    fake = ratelimit.get_redis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)

    for _ in range(3):
        await ratelimit.check_rate_limit(_request(), "t", limit=3)
    with pytest.raises(HTTPException) as exc:
        await ratelimit.check_rate_limit(_request(), "t", limit=3)
    assert exc.value.status_code == 429


async def test_limits_are_per_client(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    await ratelimit.check_rate_limit(_request("1.1.1.1"), "t", limit=1)
    await ratelimit.check_rate_limit(_request("2.2.2.2"), "t", limit=1)  # no raise


async def test_fails_open_when_redis_down(monkeypatch):
    monkeypatch.setattr(ratelimit, "get_redis", lambda: BrokenRedis())
    for _ in range(10):
        await ratelimit.check_rate_limit(_request(), "t", limit=1)  # never raises
