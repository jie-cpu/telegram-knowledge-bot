"""Trace logging — structured logging with span-based tracing.

Provides:
- Hierarchical span tracking (session → step → tool)
- Structured JSON log output
- Configurable log levels
- Console and file output
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from config.settings import LOG_DIR, settings

logger = logging.getLogger("toolagent.trace")


class TraceSpan:
    """A single span in a trace hierarchy."""

    def __init__(
        self,
        name: str,
        span_id: str | None = None,
        parent_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.span_id = span_id or uuid.uuid4().hex[:12]
        self.parent_id = parent_id
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "started"
        self.events: list[dict[str, Any]] = []

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to this span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "events": self.events,
            "metadata": {k: v for k, v in self.metadata.items() if not callable(v)},
        }


class TraceLogger:
    """Structured trace logging with span support."""

    _instance: TraceLogger | None = None

    def __new__(cls) -> TraceLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._enabled = settings.observability_trace_enabled
        self._spans: dict[str, TraceSpan] = {}
        self._active_span_stack: list[str] = []

        # Set up file handler
        if self._enabled:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            trace_log_path = LOG_DIR / "traces.jsonl"
            file_handler = logging.FileHandler(str(trace_log_path))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)

    def start_span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Start a new span, optionally as a child of the current span."""
        parent_id = self._active_span_stack[-1] if self._active_span_stack else None
        span = TraceSpan(
            name=name,
            parent_id=parent_id,
            trace_id=self._active_span_stack[0] if self._active_span_stack else None,
            metadata=metadata,
        )
        self._spans[span.span_id] = span
        self._active_span_stack.append(span.span_id)
        self._log_span_event("span_start", span)
        return span

    def end_span(self, span: TraceSpan, status: str = "ok") -> None:
        """End a span."""
        span.finish(status)
        if self._active_span_stack and self._active_span_stack[-1] == span.span_id:
            self._active_span_stack.pop()
        self._log_span_event("span_end", span)

    @contextmanager
    def span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[TraceSpan, None, None]:
        """Context manager for a span."""
        span = self.start_span(name, metadata)
        try:
            yield span
        except Exception as e:
            span.add_event("error", {"error": str(e)})
            self.end_span(span, status="error")
            raise
        else:
            self.end_span(span)

    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        duration_ms: float,
        success: bool,
    ) -> None:
        """Log a tool call with its result."""
        if not self._enabled:
            return
        entry = {
            "type": "tool_call",
            "tool": tool_name,
            "args": {k: v for k, v in tool_args.items() if not isinstance(v, str) or len(v) < 500},
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.debug(json.dumps(entry))

    def log_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
    ) -> None:
        """Log an LLM API call."""
        if not self._enabled:
            return
        entry = {
            "type": "llm_call",
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.debug(json.dumps(entry))

    def log_guardrail(self, guardrail_name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        """Log a guardrail check."""
        if not self._enabled:
            return
        entry = {
            "type": "guardrail_check",
            "guardrail": guardrail_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.debug(json.dumps(entry))

    def _log_span_event(self, event_type: str, span: TraceSpan) -> None:
        if not self._enabled:
            return
        entry = {
            "type": event_type,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "trace_id": span.trace_id,
            "name": span.name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.debug(json.dumps(entry))

    def get_trace_summary(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Get a summary of all spans in a trace."""
        if trace_id:
            spans = [s for s in self._spans.values() if s.trace_id == trace_id]
        else:
            spans = list(self._spans.values())
        return [s.to_dict() for s in spans]

    def clear(self) -> None:
        """Clear all spans."""
        self._spans.clear()
        self._active_span_stack.clear()
