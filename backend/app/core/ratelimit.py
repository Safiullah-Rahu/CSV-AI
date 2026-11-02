"""Sliding-window rate limiter (in-memory for single node, Redis for clusters)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from functools import lru_cache

from app.config import get_settings
from app.utils.errors import RateLimited


class MemoryLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float = 60.0) -> int:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return 0
            q.append(now)
            return limit - len(q)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class RedisLimiter:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url)

    def hit(self, key: str, limit: int, window_s: float = 60.0) -> int:
        now = time.time()
        rkey = f"rag:rl:{key}"
        pipe = self._r.pipeline()
        pipe.zremrangebyscore(rkey, 0, now - window_s)
        pipe.zcard(rkey)
        _, count = pipe.execute()
        if count >= limit:
            return 0
        pipe = self._r.pipeline()
        pipe.zadd(rkey, {f"{now}:{id(pipe)}": now})
        pipe.expire(rkey, int(window_s) + 1)
        pipe.execute()
        return limit - count - 1

    def reset(self) -> None:  # pragma: no cover - not used against shared redis
        pass


@lru_cache(maxsize=1)
def get_limiter() -> MemoryLimiter | RedisLimiter:
    settings = get_settings()
    if settings.redis_url and settings.event_backend == "redis":
        return RedisLimiter(settings.redis_url)
    return MemoryLimiter()


def enforce(key: str, limit: int, window_s: float = 60.0) -> None:
    if get_limiter().hit(key, limit, window_s) <= 0:
        raise RateLimited("Too many requests. Please slow down and try again shortly.")
