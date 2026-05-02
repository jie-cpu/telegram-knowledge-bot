"""Metrics collector — aggregates and reports performance metrics.

Tracks:
- Per-tool call counts, success rates, and latencies
- Step counts and completion status
- Session-level statistics
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import LOG_DIR, settings

logger = logging.getLogger("toolagent.metrics")


class MetricsCollector:
    """Collects and reports agent performance metrics."""

    _instance: MetricsCollector | None = None

    def __new__(cls) -> MetricsCollector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._enabled = settings.observability_metrics_enabled

        # Session metrics
        self._session_count = 0
        self._session_statuses: list[str] = []

        # Tool metrics
        self._tool_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._tool_latencies: dict[str, list[float]] = defaultdict(list)
        self._tool_success: dict[str, list[bool]] = defaultdict(list)

        # Step metrics
        self._step_counts: list[int] = []

        # LLM metrics
        self._llm_latencies: list[float] = []
        self._llm_costs: list[float] = []
        self._llm_tokens: list[int] = []

        # File logging
        if self._enabled:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

    def record_session(self, status: str, step_count: int) -> None:
        """Record a completed session."""
        self._session_count += 1
        self._session_statuses.append(status)
        self._step_counts.append(step_count)

    def record_tool_call(self, tool_name: str, duration_ms: float, success: bool, details: dict[str, Any] | None = None) -> None:
        """Record a tool call execution."""
        self._tool_calls[tool_name].append({
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            **(details or {}),
        })
        self._tool_latencies[tool_name].append(duration_ms)
        self._tool_success[tool_name].append(success)

    def record_llm_call(self, latency_ms: float, cost_usd: float, tokens: int) -> None:
        """Record an LLM API call."""
        self._llm_latencies.append(latency_ms)
        self._llm_costs.append(cost_usd)
        self._llm_tokens.append(tokens)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        if not self._enabled:
            return {"enabled": False}

        tool_stats = {}
        for tool_name, calls in self._tool_calls.items():
            latencies = self._tool_latencies[tool_name]
            successes = self._tool_success[tool_name]
            total = len(calls)
            tool_stats[tool_name] = {
                "total_calls": total,
                "success_rate": round(sum(successes) / total * 100, 1) if total > 0 else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 20 else 0,
            }

        return {
            "enabled": True,
            "session_count": self._session_count,
            "session_statuses": {
                status: self._session_statuses.count(status)
                for status in set(self._session_statuses)
            },
            "average_steps_per_session": (
                round(sum(self._step_counts) / len(self._step_counts), 1)
                if self._step_counts else 0
            ),
            "tool_stats": tool_stats,
            "llm_total_calls": len(self._llm_latencies),
            "llm_avg_latency_ms": (
                round(sum(self._llm_latencies) / len(self._llm_latencies), 2)
                if self._llm_latencies else 0
            ),
            "llm_total_cost_usd": round(sum(self._llm_costs), 6),
            "llm_total_tokens": sum(self._llm_tokens),
        }

    def export_json(self, filepath: str | None = None) -> str:
        """Export metrics as JSON, optionally to a file."""
        data = self.get_summary()
        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")

        return json_str

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._session_count = 0
        self._session_statuses.clear()
        self._tool_calls.clear()
        self._tool_latencies.clear()
        self._tool_success.clear()
        self._step_counts.clear()
        self._llm_latencies.clear()
        self._llm_costs.clear()
        self._llm_tokens.clear()
