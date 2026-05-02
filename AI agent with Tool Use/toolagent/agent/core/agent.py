"""ToolAgent — The main agent loop (hand-written ReAct implementation).

Implements the ReAct (Reasoning + Acting) pattern without external frameworks:
1. LLM reasons about the current state
2. LLM decides on a tool call (or final answer)
3. Execute the tool
4. Feed result back to the LLM
5. Repeat until done or limit reached

Includes step limits, timeouts, guardrails, observability, and cost tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from agent.core.llm_client import LLMClient
from agent.core.memory import ConversationMemory
from agent.tools.registry import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    MAX_STEPS_REACHED = "max_steps_reached"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


@dataclass
class AgentStep:
    step_number: int
    llm_response: Any = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: Any = None
    guardrail_check: dict[str, Any] | None = None
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > self.start_time else 0.0


@dataclass
class AgentSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: AgentStatus = AgentStatus.IDLE
    steps: list[AgentStep] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    final_response: str = ""
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    user_input: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > self.start_time else 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "steps": [
                {
                    "step": s.step_number,
                    "tool": s.tool_name,
                    "tool_input": s.tool_input,
                    "error": s.error,
                    "duration_ms": round(s.duration_ms, 2),
                }
                for s in self.steps
            ],
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "duration_ms": round(self.duration_ms, 2),
            "step_count": self.step_count,
            "final_response": self.final_response[:500] if self.final_response else "",
            "error": self.error,
        }


@dataclass
class AgentConfig:
    max_steps: int = 15
    step_timeout_seconds: int = 60
    session_timeout_seconds: int = 600
    guardrails_enabled: bool = True

    on_step_start: Callable[[int], None] | None = None
    on_step_end: Callable[[AgentStep], None] | None = None
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None
    on_session_end: Callable[[AgentSession], None] | None = None


class ToolAgent:
    """Hand-written ReAct agent loop. No LangChain/LangGraph dependency."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_client: LLMClient | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.registry = registry or create_default_registry(demo_mode=True)
        self.llm = llm_client or LLMClient()
        self.config = config or AgentConfig()
        self.memory = ConversationMemory()
        self._input_guardrails: list[Any] = []
        self._output_guardrails: list[Any] = []
        self.memory.set_system_prompt(
            ConversationMemory.create_default_system_prompt(self.registry.tool_names)
        )

    def set_guardrails(self, input_guardrails: list[Any], output_guardrails: list[Any]) -> None:
        self._input_guardrails = input_guardrails
        self._output_guardrails = output_guardrails

    async def run(self, user_input: str) -> AgentSession:
        session = AgentSession(user_input=user_input)
        session.status = AgentStatus.RUNNING
        session.start_time = time.time()

        # ── Input guardrails ──────────────────────────────────────
        if self.config.guardrails_enabled and self._input_guardrails:
            for guardrail in self._input_guardrails:
                result = await guardrail.check(user_input)
                if not result.passed:
                    session.status = AgentStatus.GUARDRAIL_BLOCKED
                    session.final_response = result.message
                    session.end_time = time.time()
                    return session

        self.memory.add_message(role="user", content=user_input)

        # ── Main reasoning loop ───────────────────────────────────
        for step_num in range(1, self.config.max_steps + 1):
            if time.time() - session.start_time > self.config.session_timeout_seconds:
                session.status = AgentStatus.TIMEOUT
                session.error = f"Session timed out after {self.config.session_timeout_seconds}s"
                break

            step = AgentStep(step_number=step_num, start_time=time.time())

            if self.config.on_step_start:
                self.config.on_step_start(step_num)

            # LLM call
            try:
                messages = self.memory.get_messages_for_openai()
                tools = self.registry.get_tools_for_openai()
                llm_response = await asyncio.wait_for(
                    self.llm.chat(messages=messages, tools=tools),
                    timeout=self.config.step_timeout_seconds,
                )
                step.llm_response = llm_response
            except asyncio.TimeoutError:
                step.error = f"Step {step_num} timed out"
                session.status = AgentStatus.TIMEOUT
                step.end_time = time.time()
                session.steps.append(step)
                break
            except Exception as e:
                step.error = f"LLM error: {e}"
                step.end_time = time.time()
                session.steps.append(step)
                continue

            # Track costs
            session.total_cost_usd += getattr(llm_response, "cost_usd", 0)
            session.total_latency_ms += getattr(llm_response, "latency_ms", 0)
            session.total_tokens += getattr(llm_response, "total_tokens", 0)

            content = llm_response.content or ""
            tool_calls = getattr(llm_response, "tool_calls", [])

            # Add assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.memory.add_message(**assistant_msg)

            # No tool calls → done
            if not tool_calls:
                session.final_response = content or "Task completed."
                session.status = AgentStatus.COMPLETED
                step.end_time = time.time()
                session.steps.append(step)
                if self.config.on_step_end:
                    self.config.on_step_end(step)
                break

            # Execute each tool call
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                tc_id = tc.get("id", "")

                step.tool_name = tool_name
                step.tool_input = tool_args

                if self.config.on_tool_call:
                    self.config.on_tool_call(tool_name, tool_args)

                # Output guardrails
                if self.config.guardrails_enabled and self._output_guardrails:
                    blocked = False
                    for guardrail in self._output_guardrails:
                        result = await guardrail.check_tool_call(tool_name, tool_args)
                        if not result.passed:
                            step.error = f"Guardrail blocked: {result.message}"
                            blocked = True
                            break
                    if blocked:
                        break

                # Execute tool
                try:
                    tool_result = await asyncio.wait_for(
                        self.registry.execute_tool(tool_name, **tool_args),
                        timeout=self.config.step_timeout_seconds,
                    )
                    step.tool_result = tool_result
                except asyncio.TimeoutError:
                    step.error = f"Tool '{tool_name}' timed out"
                    tool_result_obj = type("R", (), {"success": False, "output": None, "error": "timeout"})()
                except Exception as e:
                    step.error = f"Tool '{tool_name}' error: {e}"
                    tool_result_obj = type("R", (), {"success": False, "output": None, "error": str(e)})()
                else:
                    tool_result_obj = tool_result

                # Add tool result to memory
                result_data = {
                    "success": tool_result_obj.success,
                    "output": tool_result_obj.output,
                    "error": tool_result_obj.error,
                }
                self.memory.add_tool_result(tc_id, tool_name, result_data)

            step.end_time = time.time()
            session.steps.append(step)
            if self.config.on_step_end:
                self.config.on_step_end(step)
        else:
            if session.status == AgentStatus.RUNNING:
                session.status = AgentStatus.MAX_STEPS_REACHED
                session.final_response = (
                    f"Reached maximum of {self.config.max_steps} steps "
                    f"without completing the task."
                )

        session.end_time = time.time()
        logger.info(
            f"Session {session.session_id} | {session.status.value} | "
            f"{session.step_count} steps | ${session.total_cost_usd:.6f} | "
            f"{session.total_tokens} tokens"
        )
        if self.config.on_session_end:
            self.config.on_session_end(session)
        return session

    def reset(self) -> None:
        self.memory.clear()
