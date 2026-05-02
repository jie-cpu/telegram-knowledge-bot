"""Base tool class with JSON Schema definitions.

Every tool in the system inherits from BaseTool and provides:
- A name, description, and input/output JSON Schema
- An execute() method that performs the actual work
- Error handling and result wrapping
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolResult:
    """The result of a tool execution, including metadata."""

    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    token_count: int = 0
    tool_call_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BaseTool(ABC):
    """Abstract base class that all tools must implement.

    Each tool provides:
    - name: A unique identifier (snake_case)
    - description: What the tool does (used by the LLM for selection)
    - parameters: JSON Schema for input validation
    - execute(): The actual implementation
    """

    def __init__(self) -> None:
        self._id = f"{self.name}_{uuid.uuid4().hex[:8]}"

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier for the tool, e.g. 'calculator'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description of when to use this tool."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input parameters.

        Must follow the JSON Schema spec so it can be passed directly
        to OpenAI / Anthropic tool-use APIs.
        """
        ...

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Wraps the implementation with timing, error handling,
        and result formatting.
        """
        start = time.perf_counter()
        tool_call_id = uuid.uuid4().hex[:12]

        try:
            output = await self._run(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=round(elapsed_ms, 2),
                tool_call_id=tool_call_id,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=round(elapsed_ms, 2),
                tool_call_id=tool_call_id,
            )

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """Internal implementation — override in subclasses."""
        ...
