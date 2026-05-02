"""FastAPI application for ToolAgent.

Provides REST endpoints for agent interaction, including:
- POST /chat — send a message to the agent
- GET /tools — list available tools
- GET /metrics — view system metrics
- GET /health — health check
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.core.agent import AgentConfig, AgentSession, ToolAgent
from agent.guardrails.input_guardrails import InputContentGuardrail, InputLengthGuardrail
from agent.guardrails.output_guardrails import ToolCallValidationGuardrail
from agent.observability.cost_tracker import CostTracker
from agent.observability.metrics import MetricsCollector
from agent.observability.tracer import TraceLogger
from agent.tools.registry import create_default_registry
from api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    StepDetail,
    ToolsListResponse,
)

logger = logging.getLogger(__name__)

# ── Global agent instance ─────────────────────────────────────────
_agent: ToolAgent | None = None
_sessions: dict[str, ToolAgent] = {}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        # Startup: create default registry
        global _agent
        registry = create_default_registry(demo_mode=True)

        config = AgentConfig(
            guardrails_enabled=True,
            observability_enabled=True,
        )

        _agent = ToolAgent(
            registry=registry,
            config=config,
        )

        # Add guardrails
        input_guardrails = [
            InputLengthGuardrail(max_length=10000),
            InputContentGuardrail(),
        ]
        output_guardrails = [
            ToolCallValidationGuardrail(),
        ]
        _agent.set_guardrails(input_guardrails, output_guardrails)

        logger.info(f"ToolAgent initialized with {len(registry)} tools: {registry.tool_names}")
        yield

        # Shutdown
        global _sessions
        _sessions.clear()

    app = FastAPI(
        title="ToolAgent API",
        description="An extensible AI agent system with tool use, guardrails, and observability",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        if _agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        return HealthResponse(
            status="ok",
            version="1.0.0",
            provider=_agent.llm.provider_name,
            tool_count=len(_agent.registry),
        )

    @app.get("/tools", response_model=ToolsListResponse)
    async def list_tools():
        """List all available tools with their schemas."""
        if _agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        tools = _agent.registry.get_tools_for_openai()
        return ToolsListResponse(tools=tools, count=len(tools))

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Send a message to the agent and get a response."""
        if _agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Get or create agent session
        if request.session_id and request.session_id in _sessions:
            agent = _sessions[request.session_id]
        else:
            registry = create_default_registry(demo_mode=True)
            config = AgentConfig(max_steps=request.max_steps, guardrails_enabled=True)
            agent = ToolAgent(registry=registry, config=config)

            input_guardrails = [
                InputLengthGuardrail(max_length=10000),
                InputContentGuardrail(),
            ]
            output_guardrails = [
                ToolCallValidationGuardrail(),
            ]
            agent.set_guardrails(input_guardrails, output_guardrails)

        try:
            session: AgentSession = await agent.run(request.message)

            # Store session for continuation
            session_id = session.session_id
            _sessions[session_id] = agent

            # Build step details
            steps = [
                StepDetail(
                    step_number=s.step_number,
                    tool_name=s.tool_name,
                    tool_input=s.tool_input,
                    result=s.tool_result.output if s.tool_result else None,
                    error=s.error,
                    duration_ms=round(s.duration_ms, 2),
                )
                for s in session.steps
                if s.tool_name
            ]

            return ChatResponse(
                session_id=session_id,
                response=session.final_response or "No response generated.",
                status=session.status.value,
                steps=steps,
                total_cost_usd=round(session.total_cost_usd, 6),
                total_latency_ms=round(session.total_latency_ms, 2),
                total_tokens=session.total_tokens,
                step_count=session.step_count,
            )

        except Exception as e:
            logger.exception(f"Error in chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get system metrics."""
        metrics = MetricsCollector()
        summary = metrics.get_summary()
        return MetricsResponse(
            session_count=summary.get("session_count", 0),
            tool_stats=summary.get("tool_stats", {}),
            llm_total_calls=summary.get("llm_total_calls", 0),
            llm_total_cost_usd=summary.get("llm_total_cost_usd", 0.0),
            llm_total_tokens=summary.get("llm_total_tokens", 0),
        )

    @app.get("/costs")
    async def get_costs():
        """Get cost tracking data."""
        cost_tracker = CostTracker()
        return cost_tracker.get_summary()

    @app.get("/sessions", response_model=list)
    async def list_sessions():
        """List active sessions."""
        return [
            {
                "session_id": sid,
                "tool_count": len(a.registry),
            }
            for sid, a in _sessions.items()
        ]

    @app.delete("/sessions/{session_id}")
    async def clear_session(session_id: str):
        """Clear a specific session."""
        if session_id in _sessions:
            del _sessions[session_id]
            return {"message": f"Session {session_id} cleared"}
        raise HTTPException(status_code=404, detail="Session not found")

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
