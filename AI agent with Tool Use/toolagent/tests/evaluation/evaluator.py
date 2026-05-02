"""Evaluation harness for measuring agent performance.

Evaluates:
- Correct tool selection (did the agent choose the right tool?)
- Correct tool sequence (did the agent call tools in the right order?)
- Step efficiency (how many steps did it take?)
- Cost and latency
- Guardrail effectiveness
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent.core.agent import AgentConfig, AgentSession, ToolAgent
from agent.guardrails.input_guardrails import InputContentGuardrail, InputLengthGuardrail
from agent.guardrails.output_guardrails import ToolCallValidationGuardrail
from agent.observability.cost_tracker import CostTracker
from agent.observability.metrics import MetricsCollector
from agent.observability.tracer import TraceLogger
from agent.tools.registry import create_default_registry
from tests.evaluation.dataset import EvalExample, EvaluationDataset
from tests.evaluation.scenarios import Scenario


@dataclass
class EvalScore:
    """Score for a single evaluation example."""

    example_id: str
    prompt: str
    passed: bool = False
    tool_selection_score: float = 0.0
    sequence_score: float = 0.0
    steps_used: int = 0
    expected_tools: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)
    correct_tools: list[str] = field(default_factory=list)
    missed_tools: list[str] = field(default_factory=list)
    extra_tools: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    status: str = ""
    error: str | None = None


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    dataset_name: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_tool_selection_score: float = 0.0
    avg_sequence_score: float = 0.0
    avg_steps: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    scores: list[EvalScore] = field(default_factory=list)
    tool_accuracy: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "total_examples": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate * 100, 1),
            "avg_tool_selection_accuracy": round(self.avg_tool_selection_score * 100, 1),
            "avg_sequence_accuracy": round(self.avg_sequence_score * 100, 1),
            "avg_steps_per_task": round(self.avg_steps, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "tool_accuracy": self.tool_accuracy,
        }


class Evaluator:
    """Evaluation harness for running and scoring agent performance."""

    def __init__(self, dataset: EvaluationDataset | None = None) -> None:
        self.dataset = dataset or EvaluationDataset()
        self._results: list[EvalScore] = []

    def evaluate_example(self, score: EvalScore, session: AgentSession) -> EvalScore:
        """Score a single evaluation example against the agent's session."""
        expected = set(score.expected_tools)
        actual = [s.tool_name for s in session.steps if s.tool_name]

        score.actual_tools = actual
        score.status = session.status.value
        score.total_cost_usd = session.total_cost_usd
        score.total_latency_ms = session.total_latency_ms
        score.total_tokens = session.total_tokens
        score.steps_used = session.step_count

        # ── Tool selection score ────────────────────────────────────
        actual_set = set(actual)
        correct = expected & actual_set
        missed = expected - actual_set
        extra = actual_set - expected

        score.correct_tools = list(correct)
        score.missed_tools = list(missed)
        score.extra_tools = list(extra)

        if len(expected) == 0:
            # Guardrail test: should use no tools
            score.tool_selection_score = 1.0 if len(actual) == 0 else 0.0
        elif len(expected) == 0:
            score.tool_selection_score = 0.0
        else:
            # F1 score for tool selection
            precision = len(correct) / len(actual_set) if actual_set else 0
            recall = len(correct) / len(expected) if expected else 0
            score.tool_selection_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ── Sequence score ──────────────────────────────────────────
        expected_seq = score.expected_tools
        if expected_seq:
            # Check if tools appear in expected order
            actual_seq = [t for t in actual if t in expected_seq]
            matches = sum(1 for i, t in enumerate(actual_seq) if i < len(expected_seq) and t == expected_seq[i])
            score.sequence_score = matches / len(expected_seq) if expected_seq else 0.0
        else:
            score.sequence_score = 1.0  # No sequence expectation

        # ── Overall pass ────────────────────────────────────────────
        score.passed = (
            len(score.missed_tools) == 0
            and (len(expected) == 0 or score.tool_selection_score >= 0.5)
            and session.status not in ("error", "timeout", "guardrail_blocked")
        )

        return score

    async def run(
        self,
        dataset: EvaluationDataset | None = None,
        max_steps: int = 15,
        verbose: bool = False,
    ) -> EvaluationResult:
        """Run evaluation on a dataset."""
        ds = dataset or self.dataset
        scores: list[EvalScore] = []

        for example in ds.examples:
            if verbose:
                print(f"  Evaluating: {example.id} — {example.description}")

            # Create fresh agent for each example
            registry = create_default_registry(demo_mode=True)
            config = AgentConfig(max_steps=min(example.max_expected_steps, max_steps))
            agent = ToolAgent(registry=registry, config=config)

            # Add guardrails
            input_guardrails = [InputLengthGuardrail(), InputContentGuardrail()]
            output_guardrails = [ToolCallValidationGuardrail()]
            agent.set_guardrails(input_guardrails, output_guardrails)

            # Create score
            score = EvalScore(
                example_id=example.id,
                prompt=example.prompt,
                expected_tools=example.expected_tools,
            )

            try:
                session = await agent.run(example.prompt)
                score = self.evaluate_example(score, session)
            except Exception as e:
                score.error = str(e)
                score.status = "error"

            scores.append(score)

        # ── Aggregate results ──────────────────────────────────────
        self._results = scores
        total = len(scores)
        passed = sum(1 for s in scores if s.passed)

        if total == 0:
            return EvaluationResult()

        # Tool accuracy per tool
        tool_accuracy: dict[str, dict[str, float]] = {}
        for s in scores:
            for tool in s.expected_tools:
                if tool not in tool_accuracy:
                    tool_accuracy[tool] = {"correct": 0, "total": 0}
                tool_accuracy[tool]["total"] += 1
                if tool in s.correct_tools:
                    tool_accuracy[tool]["correct"] += 1

        tool_accuracy_summary = {}
        for tool, counts in tool_accuracy.items():
            tool_accuracy_summary[tool] = {
                "accuracy": round(counts["correct"] / counts["total"] * 100, 1) if counts["total"] > 0 else 0,
                "correct": counts["correct"],
                "total": counts["total"],
            }

        return EvaluationResult(
            dataset_name=ds.name,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total,
            avg_tool_selection_score=sum(s.tool_selection_score for s in scores) / total,
            avg_sequence_score=sum(s.sequence_score for s in scores) / total,
            avg_steps=sum(s.steps_used for s in scores) / total,
            total_cost_usd=sum(s.total_cost_usd for s in scores),
            total_tokens=sum(s.total_tokens for s in scores),
            scores=scores,
            tool_accuracy=tool_accuracy_summary,
        )

    def print_report(self, result: EvaluationResult) -> str:
        """Generate a human-readable evaluation report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"EVALUATION REPORT: {result.dataset_name}")
        lines.append("=" * 60)

        lines.append(f"\n📊 Overall Results:")
        lines.append(f"  Total Examples:  {result.total}")
        lines.append(f"  Passed:          {result.passed}")
        lines.append(f"  Failed:          {result.failed}")
        lines.append(f"  Pass Rate:       {result.pass_rate * 100:.1f}%")
        lines.append(f"  Avg Selection:   {result.avg_tool_selection_score * 100:.1f}%")
        lines.append(f"  Avg Sequence:    {result.avg_sequence_score * 100:.1f}%")
        lines.append(f"  Avg Steps:       {result.avg_steps:.1f}")
        lines.append(f"  Total Cost:      ${result.total_cost_usd:.6f}")
        lines.append(f"  Total Tokens:    {result.total_tokens}")

        lines.append(f"\n🔧 Tool Accuracy:")
        for tool, stats in sorted(result.tool_accuracy.items()):
            lines.append(f"  {tool}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")

        lines.append(f"\n📋 Detailed Results:")
        for score in result.scores:
            status = "✅" if score.passed else "❌"
            selection = f"{score.tool_selection_score * 100:.0f}%"
            lines.append(f"  {status} {score.example_id}: selection={selection}, "
                         f"steps={score.steps_used}, tools={score.actual_tools}")

        return "\n".join(lines)

    def export_json(self, result: EvaluationResult, filepath: str) -> str:
        """Export evaluation results as JSON."""
        data = result.summary()
        data["scores"] = [
            {
                "id": s.example_id,
                "passed": s.passed,
                "tool_selection_score": s.tool_selection_score,
                "sequence_score": s.sequence_score,
                "steps": s.steps_used,
                "expected": s.expected_tools,
                "actual": s.actual_tools,
                "cost": s.total_cost_usd,
                "tokens": s.total_tokens,
            }
            for s in result.scores
        ]
        json_str = json.dumps(data, indent=2)
        import pathlib
        pathlib.Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str
