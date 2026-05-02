"""Token-bucket rate limiter per user."""

import time
from collections import defaultdict


class TokenBucket:
    """Simple token-bucket rate limiter.

    Each user (by telegram_id) gets at most *capacity* messages per
    *window_seconds* sliding window.  Intended for in-process use; for
    multi-process deployments swap in a Redis-backed implementation.
    """

    def __init__(self, capacity: int = 10, window_seconds: int = 60) -> None:
        self.capacity = capacity
        self.window = window_seconds
        self._buckets: dict[int, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        timestamps = self._buckets[user_id]
        cutoff = now - self.window
        # prune expired timestamps
        self._buckets[user_id] = [t for t in timestamps if t > cutoff]
        bucket = self._buckets[user_id]
        if len(bucket) >= self.capacity:
            return False
        bucket.append(now)
        return True

    def remaining(self, user_id: int) -> int:
        now = time.monotonic()
        cutoff = now - self.window
        bucket = [t for t in self._buckets[user_id] if t > cutoff]
        self._buckets[user_id] = bucket
        return max(0, self.capacity - len(bucket))

    def reset(self, user_id: int) -> None:
        self._buckets[user_id].clear()
