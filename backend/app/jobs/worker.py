"""Job worker and scheduler.

    python -m app.jobs.worker            # standalone worker process (production)

With ``JOB_BACKEND=embedded`` the API process starts the same loop in
background threads at startup (single-node and development deployments).

The scheduler tick (scheduled syncs, stale-lease recovery, retention) is
guarded by a PostgreSQL advisory lock so only one instance runs it, no
matter how many workers are deployed.
"""

from __future__ import annotations

import os
import select as _select
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg

from app.config import get_settings
from app.db.session import get_engine
from app.jobs import queue
from app.utils.logging import configure_logging, get_logger

log = get_logger("worker")
SCHEDULER_LOCK_ID = 7_310_442


class Worker:
    def __init__(self, concurrency: int | None = None) -> None:
        settings = get_settings()
        self.concurrency = concurrency or settings.worker_concurrency
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self.pool = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="job")
        self.stop_event = threading.Event()
        self.inflight = threading.Semaphore(self.concurrency)
        self._threads: list[threading.Thread] = []

    # ----------------------------------------------------------------- lifecycle
    def start_background(self) -> None:
        import app.jobs.handlers  # noqa: F401 — register handlers

        for target, name in ((self._poll_loop, "job-poller"), (self._scheduler_loop, "job-scheduler")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("worker started id=%s concurrency=%s", self.worker_id, self.concurrency)

    def stop(self) -> None:
        self.stop_event.set()
        self.pool.shutdown(wait=False, cancel_futures=False)

    # ---------------------------------------------------------------------- loops
    def _listen_conn(self) -> psycopg.Connection | None:
        try:
            url = get_engine().url.set(drivername="postgresql").render_as_string(hide_password=False)
            conn = psycopg.connect(url, autocommit=True)
            conn.execute("LISTEN rag_jobs")
            return conn
        except Exception:  # noqa: BLE001 — fall back to polling
            log.exception("LISTEN unavailable; polling only")
            return None

    def _poll_loop(self) -> None:
        conn = self._listen_conn()
        while not self.stop_event.is_set():
            try:
                self._drain()
            except Exception:  # noqa: BLE001
                log.exception("poll failed")
            # Sleep until a NOTIFY arrives or 2 s pass.
            if conn is not None:
                try:
                    ready, _, _ = _select.select([conn.fileno()], [], [], 2.0)
                    if ready:
                        for _ in conn.notifies(timeout=0):
                            pass
                except Exception:  # noqa: BLE001
                    conn = None
            else:
                self.stop_event.wait(2.0)

    def _drain(self) -> None:
        while not self.stop_event.is_set() and self.inflight.acquire(blocking=False):
            claimed = queue.claim(None, self.worker_id, limit=1)
            if not claimed:
                self.inflight.release()
                return
            self.pool.submit(self._run, claimed[0])

    def _run(self, claimed: tuple) -> None:  # type: ignore[type-arg]
        try:
            queue._execute(*claimed)
        finally:
            self.inflight.release()

    def _scheduler_loop(self) -> None:
        settings = get_settings()
        if not settings.scheduler_enabled:
            return
        while not self.stop_event.wait(settings.scheduler_interval_seconds):
            try:
                self.scheduler_tick()
            except Exception:  # noqa: BLE001
                log.exception("scheduler tick failed")

    @staticmethod
    def scheduler_tick() -> None:
        from sqlalchemy import text

        from app.services.scheduling import run_due_schedules, run_retention

        with get_engine().connect() as conn:
            got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": SCHEDULER_LOCK_ID}).scalar()
            if not got:
                return
            try:
                requeued = queue.requeue_stale()
                if requeued:
                    log.warning("requeued %s stale jobs", requeued)
                run_due_schedules()
                run_retention()
                from app.llm.catalog import refresh_all

                refresh_all()
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": SCHEDULER_LOCK_ID})
                conn.commit()


_embedded: Worker | None = None


def start_embedded() -> Worker:
    global _embedded
    if _embedded is None:
        _embedded = Worker()
        _embedded.start_background()
    return _embedded


def stop_embedded() -> None:
    global _embedded
    if _embedded is not None:
        _embedded.stop()
        _embedded = None


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    worker = Worker()
    worker.start_background()

    def _graceful(*_args: object) -> None:
        log.info("shutting down worker")
        worker.stop()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    while not worker.stop_event.is_set():
        time.sleep(1)


if __name__ == "__main__":
    main()
