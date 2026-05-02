"""Tests for the evaluation harness."""

import pytest

from tests.evaluation.dataset import EvalExample, EvaluationDataset, create_golden_dataset
from tests.evaluation.evaluator import EvalScore, Evaluator
from tests.evaluation.scenarios import BUSINESS_ANALYSIS, Scenario


class TestDataset:
    """Tests for the evaluation dataset."""

    def test_golden_dataset_creation(self) -> None:
        dataset = create_golden_dataset()
        assert len(dataset.examples) > 0
        assert "calc_001" in [e.id for e in dataset.examples]

    def test_filter_by_tag(self) -> None:
        dataset = create_golden_dataset()
        calc_examples = dataset.filter_by_tag("calculator")
        assert len(calc_examples) > 0
        assert all("calculator" in e.tags for e in calc_examples)

    def test_tool_coverage(self) -> None:
        dataset = create_golden_dataset()
        coverage = dataset.tool_coverage
        assert "calculator" in coverage
        assert coverage["calculator"] >= 2

    def test_summary(self) -> None:
        dataset = create_golden_dataset()
        summary = dataset.summary()
        assert summary["total_examples"] > 0
        assert "calculator" in summary["tool_coverage"]


class TestEvaluator:
    """Tests for the evaluation harness."""

    def test_score_calculation(self) -> None:
        score = EvalScore(
            example_id="test_001",
            prompt="test",
            expected_tools=["calculator", "weather"],
            actual_tools=["calculator", "weather"],
            correct_tools=["calculator", "weather"],
        )
        assert score.tool_selection_score > 0

    def test_score_with_extra_tool(self) -> None:
        score = EvalScore(
            example_id="test_002",
            prompt="test",
            expected_tools=["calculator"],
            actual_tools=["calculator", "weather"],
            correct_tools=["calculator"],
            extra_tools=["weather"],
        )
        # Precision = 1/2, Recall = 1/1, F1 = 0.666...
        assert 0.6 < score.tool_selection_score < 0.7

    def test_score_with_missing_tool(self) -> None:
        score = EvalScore(
            example_id="test_003",
            prompt="test",
            expected_tools=["calculator", "weather"],
            actual_tools=["calculator"],
            correct_tools=["calculator"],
            missed_tools=["weather"],
        )
        # Precision = 1/1, Recall = 1/2, F1 = 0.666...
        assert 0.6 < score.tool_selection_score < 0.7


class TestScenarios:
    """Tests for evaluation scenarios."""

    def test_business_analysis_scenario(self) -> None:
        assert BUSINESS_ANALYSIS.name == "business_analysis"
        assert len(BUSINESS_ANALYSIS.expected_steps) >= 2
        assert "database" in BUSINESS_ANALYSIS.tags

    def test_scenario_to_dict(self) -> None:
        scenario = Scenario(name="test", description="test", prompt="test", tags=["test"])
        data = scenario.to_dict()
        assert data["name"] == "test"
        assert data["tags"] == ["test"]
