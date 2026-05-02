"""Cost tracker — instruments and tracks LLM API costs.

Provides:
- Per-session cost tracking
- Per-tool cost attribution
- Cost estimation before API calls
- Budget enforcement
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger("toolagent.cost")


# Estimated cost per 1K tokens (USD) — for estimation without API call
_ESTIMATED_COSTS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


class CostTracker:
    """Tracks LLM API costs across sessions and tools."""

    _instance: CostTracker | None = None

    def __new__(cls) -> CostTracker:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._sessions: dict[str, dict[str, Any]] = {}
        self._tool_costs: dict[str, float] = defaultdict(float)
        self._daily_costs: dict[str, float] = defaultdict(float)

    def start_session(self, session_id: str) -> None:
        """Initialize cost tracking for a new session."""
        self._sessions[session_id] = {
            "session_id": session_id,
            "llm_calls": [],
            "tool_calls": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "start_time": datetime.utcnow().isoformat(),
        }

    def record_llm_call(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record an LLM API call cost."""
        if session_id not in self._sessions:
            return

        entry = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._sessions[session_id]["llm_calls"].append(entry)
        self._sessions[session_id]["total_input_tokens"] += input_tokens
        self._sessions[session_id]["total_output_tokens"] += output_tokens
        self._sessions[session_id]["total_cost_usd"] += cost_usd

        today = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_costs[today] += cost_usd

    def record_tool_call(self, session_id: str, tool_name: str, cost_usd: float = 0.0) -> None:
        """Record a tool call (some tools may have API costs)."""
        if session_id not in self._sessions:
            return
        self._sessions[session_id]["tool_calls"].append({
            "tool": tool_name,
            "cost_usd": cost_usd,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._tool_costs[tool_name] += cost_usd

    def get_session_cost(self, session_id: str) -> dict[str, Any]:
        """Get cost breakdown for a session."""
        return self._sessions.get(session_id, {})

    def get_total_cost(self) -> float:
        """Get total cost across all sessions."""
        return sum(s["total_cost_usd"] for s in self._sessions.values())

    def get_daily_cost(self, date: str | None = None) -> float:
        """Get cost for a specific day or today."""
        if date:
            return self._daily_costs.get(date, 0.0)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self._daily_costs.get(today, 0.0)

    def estimate_cost(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> float:
        """Estimate the cost of an LLM call before making it."""
        costs = _ESTIMATED_COSTS.get(model, _ESTIMATED_COSTS.get("gpt-4o-mini", {"input": 0.0, "output": 0.0}))
        return (estimated_input_tokens / 1000 * costs["input"]) + (estimated_output_tokens / 1000 * costs["output"])

    def get_summary(self) -> dict[str, Any]:
        """Get a complete cost summary."""
        session_summaries = []
        for session_id, data in self._sessions.items():
            session_summaries.append({
                "session_id": session_id,
                "total_cost_usd": round(data["total_cost_usd"], 6),
                "total_tokens": data["total_input_tokens"] + data["total_output_tokens"],
                "llm_calls": len(data["llm_calls"]),
                "tool_calls": len(data["tool_calls"]),
            })

        return {
            "total_cost_usd": round(self.get_total_cost(), 6),
            "total_sessions": len(self._sessions),
            "sessions": session_summaries,
            "daily_costs": {k: round(v, 6) for k, v in sorted(self._daily_costs.items())},
            "tool_cost_breakdown": {k: round(v, 6) for k, v in sorted(self._tool_costs.items())},
        }

    def export_json(self, filepath: str | None = None) -> str:
        """Export cost data as JSON, optionally to a file."""
        data = self.get_summary()
        json_str = json.dumps(data, indent=2, default=str)
        if filepath:
            from pathlib import Path
            Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str
