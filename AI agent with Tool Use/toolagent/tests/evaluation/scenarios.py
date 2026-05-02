"""Multi-step evaluation scenarios for testing correct tool sequences.

Scenarios test the agent's ability to:
1. Choose the right tool for each step
2. Maintain correct ordering of tool calls
3. Handle dependencies between steps
4. Recover from errors and try alternatives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """A reusable evaluation scenario with expected behavior."""

    name: str
    description: str
    prompt: str
    expected_steps: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "expected_steps": self.expected_steps,
            "tags": self.tags,
        }


@dataclass
class MultiStepScenario:
    """A complex multi-step scenario combining multiple tools."""

    name: str
    description: str
    prompt: str
    expected_flow: list[dict[str, Any]] = field(default_factory=list)
    validation_rules: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ── Pre-built scenarios ───────────────────────────────────────────

BUSINESS_ANALYSIS = Scenario(
    name="business_analysis",
    description="Query database and analyze product data",
    prompt="Get all electronics products from the database and calculate their average price",
    expected_steps=[
        {"tool": "database_query", "input_contains": "products"},
        {"tool": "calculator", "input_contains": "average"},
    ],
    tags=["database", "calculator", "analysis"],
)

WEATHER_RESEARCH = Scenario(
    name="weather_research",
    description="Get weather and search for activities",
    prompt="What's the weather in Paris? Also search for popular tourist attractions there.",
    expected_steps=[
        {"tool": "weather", "input_contains": "Paris"},
        {"tool": "web_search", "input_contains": "Paris"},
    ],
    tags=["weather", "search", "travel"],
)

FULL_WORKFLOW = Scenario(
    name="full_workflow",
    description="Complete business workflow: query, compute, summarize, send",
    prompt="Get the total value of all orders in the database, write the result to a file, and email it to admin@example.com",
    expected_steps=[
        {"tool": "database_query"},
        {"tool": "calculator"},
        {"tool": "file_manager"},
        {"tool": "send_email"},
    ],
    tags=["workflow", "multi_tool", "advanced"],
)

CODE_ANALYSIS = Scenario(
    name="code_analysis",
    description="Write and execute Python code for data analysis",
    prompt="Write Python code to calculate the sum of numbers from 1 to 100, then execute it",
    expected_steps=[
        {"tool": "code_executor", "input_contains": "sum"},
    ],
    tags=["code", "python"],
)
