"""Organisation-scoped pub/sub used for real-time UI updates.

Events are persisted in ``activity_events`` (the source of truth, replayable
via ``Last-Event-ID``) and then published to live subscribers *after* the
transaction commits, so subscribers never see events for rolled-back work.

Backends:
* ``memory`` — single process (dev, tests, thread job runner).
* ``redis`` — multi-process (API replicas + Celery workers).
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("events")


@dataclass
class _Subscriber:
    queue: asyncio.Queue[dict[str, Any]]
    loop: asyncio.AbstractEventLoop


class MemoryBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[_Subscriber]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, org_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs.get(org_id, ()))
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(_offer, sub.queue, event)
            except RuntimeError:  # loop closed
                continue

    async def subscribe(self, org_id: str) -> AsyncIterator[dict[str, Any]]:
        sub = _Subscriber(asyncio.Queue(maxsize=1000), asyncio.get_running_loop())
        with self._lock:
            self._subs[org_id].append(sub)
        try:
            while True:
                yield await sub.queue.get()
        finally:
            with self._lock:
                if sub in self._subs.get(org_id, []):
                    self._subs[org_id].remove(sub)


def _offer(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    if queue.full():  # slow consumer: drop oldest, the client can resync from the DB
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)


class RedisBus:
    def __init__(self, url: str) -> None:
        import redis

        self._url = url
        self._client = redis.Redis.from_url(url)

    @staticmethod
    def _channel(org_id: str) -> str:
        return f"rag:events:{org_id}"

    def publish(self, org_id: str, event: dict[str, Any]) -> None:
        try:
            self._client.publish(self._channel(org_id), json.dumps(event, default=str))
        except Exception:  # noqa: BLE001 — events are best-effort; DB row is authoritative
            log.exception("redis publish failed")

    async def subscribe(self, org_id: str) -> AsyncIterator[dict[str, Any]]:
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(self._url)
        pubsub = client.pubsub()
        await pubsub.subscribe(self._channel(org_id))
        try:
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe()
            await client.aclose()


@lru_cache(maxsize=1)
def get_bus() -> MemoryBus | RedisBus:
    settings = get_settings()
    if settings.event_backend == "redis" and settings.redis_url:
        return RedisBus(settings.redis_url)
    return MemoryBus()


def publish(org_id: uuid.UUID | str, event: dict[str, Any]) -> None:
    get_bus().publish(str(org_id), event)
