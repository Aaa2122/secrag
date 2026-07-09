"""Fixed-window rate limiting on Redis. Fail-open: a broken limiter must not
take the API down with it — we log and let traffic through instead.
"""

import logging
import time

from fastapi import HTTPException, Request

from secrag.config import get_settings

log = logging.getLogger(__name__)

_redis = None
_warned = False


def get_redis():
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(
            get_settings().redis_url, socket_connect_timeout=1, socket_timeout=1
        )
    return _redis


async def check_rate_limit(request: Request, scope: str, limit: int, window_s: int = 60) -> None:
    """Raise 429 when `limit` requests per `window_s` is exceeded for this client."""
    global _warned
    client = request.client.host if request.client else "unknown"
    key = f"rl:{scope}:{client}:{int(time.time() // window_s)}"
    try:
        r = get_redis()
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, window_s)
        _warned = False
    except HTTPException:
        raise
    except Exception:
        if not _warned:
            log.warning("rate limiter unavailable, failing open", exc_info=True)
            _warned = True
        return
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded: {limit} requests per {window_s}s for {scope}",
        )
