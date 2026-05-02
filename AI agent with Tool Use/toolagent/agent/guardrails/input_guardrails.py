"""Input guardrails — validate and sanitize user input before processing."""

from __future__ import annotations

import re
from typing import Any

from agent.guardrails.base import BaseGuardrail, GuardrailResult


class InputLengthGuardrail(BaseGuardrail):
    """Ensures user input doesn't exceed maximum length."""

    def __init__(self, max_length: int = 10000) -> None:
        super().__init__(name="input_length_guardrail")
        self._max_length = max_length

    async def check(self, user_input: str) -> GuardrailResult:
        if len(user_input) > self._max_length:
            return GuardrailResult(
                passed=False,
                message=f"Input exceeds maximum length of {self._max_length} characters (got {len(user_input)}).",
                details={"input_length": len(user_input), "max_length": self._max_length},
            )
        if len(user_input) == 0:
            return GuardrailResult(
                passed=False,
                message="Input cannot be empty.",
                details={"input_length": 0},
            )
        return GuardrailResult(
            passed=True,
            details={"input_length": len(user_input)},
        )


class InputContentGuardrail(BaseGuardrail):
    """Blocks inputs containing dangerous patterns or injection attempts."""

    def __init__(self) -> None:
        super().__init__(name="input_content_guardrail")
        self._blocked_patterns = [
            # Prompt injection attempts
            r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
            r"(?i)forget\s+(everything|all|previous)",
            r"(?i)system\s+prompt:?",
            r"(?i)you\s+are\s+(not\s+)?(an?\s+)?(AI|assistant|GPT)",
            r"(?i)print\s+(your|the)\s+(instructions|prompt|system)",
            r"(?i)output\s+(your|the)\s+(instructions|prompt|system)",
            r"(?i)reveal\s+(your|the)\s+(instructions|prompt|system)",
            r"(?i)act\s+as\s+(if\s+)?you\s+are",
            r"(?i)do\s+not\s+follow",
            r"(?i)new\s+instructions?:",
            # Dangerous code patterns
            r"(?i)(rm\s+-rf|format\s+|del\s+/[fsq])",
            r"(?i)(DROP|TRUNCATE)\s+TABLE",
            r"(?i)(GRANT|REVOKE)\s+ALL\s+PRIVILEGES",
        ]

    async def check(self, user_input: str) -> GuardrailResult:
        for pattern in self._blocked_patterns:
            match = re.search(pattern, user_input)
            if match:
                return GuardrailResult(
                    passed=False,
                    message=f"Input contains blocked content pattern: '{match.group()[:50]}'",
                    details={"matched_pattern": match.group()[:100]},
                )
        return GuardrailResult(passed=True)


class PIIGuardrail(BaseGuardrail):
    """Detects and blocks personally identifiable information."""

    def __init__(self, block_credit_cards: bool = True, block_ssn: bool = True) -> None:
        super().__init__(name="pii_guardrail")
        self._patterns: list[tuple[str, str]] = []
        if block_credit_cards:
            self._patterns.append((
                r"\b(?:\d[ -]*?){13,16}\b",
                "Potential credit card number detected",
            ))
        if block_ssn:
            self._patterns.append((
                r"\b\d{3}-\d{2}-\d{4}\b",
                "Potential Social Security Number detected",
            ))

    async def check(self, user_input: str) -> GuardrailResult:
        for pattern, message in self._patterns:
            if re.search(pattern, user_input):
                return GuardrailResult(
                    passed=False,
                    message=message,
                    details={"blocked_by": "pii_detection"},
                )
        return GuardrailResult(passed=True)
