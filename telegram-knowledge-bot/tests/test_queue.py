"""Tests for the message queue and rate limiter."""

import asyncio
import time

import pytest

from src.queue.message_queue import MessageQueue, Priority
from src.queue.rate_limiter import TokenBucket
from src.models.schemas import IncomingMessage, MessageType


# ---- Rate Limiter ----


class TestTokenBucket:
    def test_allows_within_limit(self) -> None:
        bucket = TokenBucket(capacity=5, window_seconds=10)
        for _ in range(5):
            assert bucket.is_allowed(1)
        assert bucket.remaining(1) == 0

    def test_blocks_when_exceeded(self) -> None:
        bucket = TokenBucket(capacity=3, window_seconds=60)
        for _ in range(3):
            bucket.is_allowed(1)
        assert not bucket.is_allowed(1)

    def test_remaining(self) -> None:
        bucket = TokenBucket(capacity=5, window_seconds=60)
        assert bucket.remaining(1) == 5
        bucket.is_allowed(1)
        assert bucket.remaining(1) == 4

    def test_reset(self) -> None:
        bucket = TokenBucket(capacity=3, window_seconds=60)
        for _ in range(3):
            bucket.is_allowed(1)
        bucket.reset(1)
        assert bucket.remaining(1) == 3

    def test_per_user_isolation(self) -> None:
        bucket = TokenBucket(capacity=2, window_seconds=60)
        bucket.is_allowed(1)
        bucket.is_allowed(1)
        assert not bucket.is_allowed(1)  # user 1 exhausted
        assert bucket.is_allowed(2)  # user 2 still allowed


# ---- Message Queue ----


class TestMessageQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue_priority(self) -> None:
        processed: list[str] = []

        async def processor(msg: IncomingMessage) -> None:
            processed.append(msg.id)

        queue = MessageQueue(worker_count=2)
        queue.set_processor(processor)

        msg_high = _make_msg("high", MessageType.TEXT)
        msg_low = _make_msg("low", MessageType.VOICE)

        await queue.enqueue(msg_low, Priority.LOW)
        await queue.enqueue(msg_high, Priority.HIGH)

        await queue.start()
        await asyncio.sleep(0.5)
        await queue.stop(timeout=2.0)

        # High priority should be processed before low
        assert len(processed) == 2

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        attempt_count = 0

        async def failing_processor(msg: IncomingMessage) -> None:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Temporary error")

        queue = MessageQueue(worker_count=1)
        queue.set_processor(failing_processor)

        msg = _make_msg("retry-test", MessageType.TEXT)
        await queue.enqueue(msg)
        await queue.start()
        await asyncio.sleep(1.0)
        await queue.stop(timeout=2.0)

        assert attempt_count >= 2, f"Expected at least 2 attempts, got {attempt_count}"

    @pytest.mark.asyncio
    async def test_pending_count(self) -> None:
        async def slow_processor(msg: IncomingMessage) -> None:
            await asyncio.sleep(1.0)

        queue = MessageQueue(worker_count=1)
        queue.set_processor(slow_processor)

        for i in range(3):
            msg = _make_msg(f"pending-{i}", MessageType.TEXT)
            await queue.enqueue(msg)

        assert queue.pending_count == 3


def _make_msg(suffix: str, msg_type: MessageType) -> IncomingMessage:
    return IncomingMessage(
        telegram_message_id=hash(suffix) % 100000,
        telegram_user_id=42,
        chat_id=-1000,
        message_type=msg_type,
        text=f"test message {suffix}",
    )
