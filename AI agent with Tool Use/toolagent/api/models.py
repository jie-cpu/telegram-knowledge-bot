"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    message: str = Field(..., description="The user's input message", min_length=1, max_length=10000)
    max_steps: int = Field(default=15, description="Maximum agent reasoning steps", ge=1, le=50)
    session_id: str | None = Field(default=None, description="Existing session ID to continue")
    stream: bool = Field(default=False, description="Enable streaming response")


class ToolCallLog(BaseModel):
    """Log of a single tool call in a step."""

    step: int
    tool: str
    tool_input: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0


class StepDetail(BaseModel):
    """Details of a single agent step."""

    step_number: int
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    session_id: str
    response: str
    status: str
    steps: list[StepDetail] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    step_count: int = 0


class SessionInfo(BaseModel):
    """Information about a session."""

    session_id: str
    status: str
    step_count: int
    total_cost_usd: float
    total_tokens: int


class MetricsResponse(BaseModel):
    """System metrics response."""

    session_count: int
    tool_stats: dict[str, Any]
    llm_total_calls: int
    llm_total_cost_usd: float
    llm_total_tokens: int


class ToolsListResponse(BaseModel):
    """List of available tools."""

    tools: list[dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    provider: str = ""
    tool_count: int = 0


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
