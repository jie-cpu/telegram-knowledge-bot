"""Async in-process message queue with priority, retry, and back-pressure.

For a single-process bot this is sufficient.  Move to Redis / RabbitMQ when
scaling horizontally.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from src.models.schemas import IncomingMessage

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    HIGH = 0  # text messages, direct replies
    MEDIUM = 1  # voice/images
    LOW = 2  # background enrichment


class RetryPolicy:
    """Exponential-backoff retry configuration."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay_s
        self.max_delay = max_delay_s


@dataclass
class QueueItem:
    message: IncomingMessage
    priority: Priority = Priority.MEDIUM
    retries_left: int = 3
    enqueued_at: float = field(default_factory=time.monotonic)
    last_error: str | None = None


ProcessorFn = Callable[[IncomingMessage], Coroutine[Any, Any, None]]


class MessageQueue:
    """Async priority queue with bounded concurrency and retry."""

    def __init__(
        self,
        worker_count: int = 2,
        retry_policy: RetryPolicy | None = None,
        max_size: int = 500,
    ) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, QueueItem]] = (
            asyncio.PriorityQueue(maxsize=max_size)
        )
        self._semaphore = asyncio.Semaphore(worker_count)
        self._retry = retry_policy or RetryPolicy()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._counter = 0  # tie-breaker for same-priority items
        self._processor: ProcessorFn | None = None
        self._running = False

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def set_processor(self, fn: ProcessorFn) -> None:
        self._processor = fn

    async def enqueue(
        self,
        message: IncomingMessage,
        priority: Priority = Priority.MEDIUM,
    ) -> None:
        item = QueueItem(message=message, priority=priority)
        self._counter += 1
        await self._queue.put((priority.value, self._counter, item))
        logger.debug(
            "Enqueued message %s (priority=%s, pending=%d)",
            message.id[:8],
            priority.name,
            self.pending_count,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._processor is None:
            raise RuntimeError("set_processor() must be called before start()")
        for _ in range(self._semaphore._value):  # noqa: SLF001
            worker = asyncio.create_task(self._worker_loop())
            self._worker_tasks.append(worker)
        logger.info("Started %d queue workers", len(self._worker_tasks))

    async def stop(self, timeout: float = 10.0) -> None:
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.wait(self._worker_tasks, timeout=timeout)
        self._worker_tasks.clear()
        logger.info("Queue workers stopped")

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                _, _, item = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            async with self._semaphore:
                try:
                    await self._process_with_retry(item)
                except Exception:
                    logger.exception(
                        "Unhandled error processing message %s", item.message.id[:8]
                    )
                finally:
                    self._queue.task_done()

    async def _process_with_retry(self, item: QueueItem) -> None:
        assert self._processor is not None
        attempt = 0
        while attempt <= item.retries_left:
            try:
                await self._processor(item.message)
                return
            except Exception as exc:
                attempt += 1
                item.last_error = str(exc)
                if attempt > item.retries_left:
                    logger.error(
                        "Message %s failed after %d retries: %s",
                        item.message.id[:8],
                        attempt - 1,
                        exc,
                    )
                    return
                delay = min(
                    self._retry.base_delay * (2 ** (attempt - 1)),
                    self._retry.max_delay,
                )
                logger.warning(
                    "Retry %d/%d for message %s in %.1fs: %s",
                    attempt,
                    item.retries_left,
                    item.message.id[:8],
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
