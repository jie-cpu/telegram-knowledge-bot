"""Base guardrail class for input validation and business rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    message: str = ""
    details: dict[str, Any] | None = None


class BaseGuardrail(ABC):
    """Abstract base class for all guardrails."""

    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def check(self, user_input: str) -> GuardrailResult:
        """Check user input against this guardrail."""
        ...

    async def check_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> GuardrailResult:
        """Check a tool call against this guardrail (default: pass)."""
        return GuardrailResult(passed=True)
