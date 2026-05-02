"""Output guardrails — validate tool calls and enforce business rules."""

from __future__ import annotations

import time
from typing import Any

from agent.guardrails.base import BaseGuardrail, GuardrailResult


class ToolCallValidationGuardrail(BaseGuardrail):
    """Validates tool call parameters against business rules."""

    def __init__(self) -> None:
        super().__init__(name="tool_call_validation")
        self._rules: dict[str, list[dict[str, Any]]] = {
            "code_executor": [
                {"field": "code", "check": "no_dangerous_imports",
                 "pattern": r"(import\s+os|import\s+subprocess|import\s+shutil|import\s+ctypes|import\s+socket)"},
            ],
            "send_email": [
                {"field": "to", "check": "valid_email",
                 "message": "All recipients must have valid email addresses"},
            ],
            "file_manager": [
                {"field": "path", "check": "no_path_traversal",
                 "pattern": r"\.\./"},
            ],
            "database_query": [
                {"field": "query", "check": "no_destructive_sql",
                 "pattern": r"(?i)\b(DROP|TRUNCATE|DELETE|ALTER|INSERT|UPDATE)\s"},
            ],
        }

    async def check(self, user_input: str) -> GuardrailResult:
        """Input check always passes — this guardrail validates tool calls."""
        return GuardrailResult(passed=True)

    async def check_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> GuardrailResult:
        if tool_name not in self._rules:
            return GuardrailResult(passed=True)

        for rule in self._rules[tool_name]:
            field = rule["field"]
            value = str(tool_args.get(field, ""))

            if rule["check"] == "no_dangerous_imports":
                import re
                if re.search(rule["pattern"], value):
                    return GuardrailResult(
                        passed=False,
                        message=f"Tool call blocked: {rule.get('message', 'Dangerous code detected')}",
                        details={"tool": tool_name, "field": field, "rule": rule["check"]},
                    )

            elif rule["check"] == "valid_email":
                import re
                invalid = [addr for addr in tool_args.get("to", []) if "@" not in str(addr)]
                if invalid:
                    return GuardrailResult(
                        passed=False,
                        message=f"Invalid email addresses: {invalid}",
                        details={"tool": tool_name, "invalid_emails": invalid},
                    )

            elif rule["check"] == "no_path_traversal":
                if ".." in value:
                    return GuardrailResult(
                        passed=False,
                        message="Path traversal detected in file_manager call",
                        details={"tool": tool_name, "path": value},
                    )

            elif rule["check"] == "no_destructive_sql":
                import re
                if re.search(rule["pattern"], value):
                    return GuardrailResult(
                        passed=False,
                        message="Destructive SQL operations are not allowed",
                        details={"tool": tool_name, "query_preview": value[:100]},
                    )

        return GuardrailResult(passed=True)


class RateLimitGuardrail(BaseGuardrail):
    """Rate-limits tool calls to prevent abuse."""

    def __init__(self, max_calls_per_minute: int = 30) -> None:
        super().__init__(name="rate_limit_guardrail")
        self._max_calls = max_calls_per_minute
        self._call_timestamps: list[float] = []

    async def check(self, user_input: str) -> GuardrailResult:
        now = time.time()
        # Remove timestamps older than 1 minute
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        self._call_timestamps.append(now)

        if len(self._call_timestamps) > self._max_calls:
            return GuardrailResult(
                passed=False,
                message=f"Rate limit exceeded: maximum {self._max_calls} requests per minute",
                details={"current_count": len(self._call_timestamps), "max": self._max_calls},
            )
        return GuardrailResult(passed=True)


class BusinessRuleGuardrail(BaseGuardrail):
    """Enforces domain-specific business rules.

    Configure with custom rules for specific use cases.
    """

    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        super().__init__(name="business_rule_guardrail")
        self._rules = rules or []

    async def check_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> GuardrailResult:
        for rule in self._rules:
            if rule.get("tool") and rule["tool"] != tool_name:
                continue

            field = rule.get("field")
            if field:
                value = tool_args.get(field)
                if rule.get("condition") == "max_value" and isinstance(value, (int, float)):
                    max_val = rule.get("max", float("inf"))
                    if value > max_val:
                        return GuardrailResult(
                            passed=False,
                            message=rule.get("message", f"Value exceeds maximum of {max_val}"),
                            details={"field": field, "value": value, "max": max_val},
                        )

                elif rule.get("condition") == "allowed_values" and value is not None:
                    allowed = rule.get("values", [])
                    if value not in allowed:
                        return GuardrailResult(
                            passed=False,
                            message=rule.get("message", f"Value '{value}' not allowed. Allowed: {allowed}"),
                            details={"field": field, "value": value, "allowed": allowed},
                        )

        return GuardrailResult(passed=True)
