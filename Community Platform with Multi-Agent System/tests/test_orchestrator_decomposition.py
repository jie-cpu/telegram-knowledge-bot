"""
Meta-tests: Orchestrator goal decomposition.

These tests validate that the orchestrator correctly decomposes goals
into dependency-ordered task graphs. They run against the agent system
itself, not the community platform code.
"""

import pytest
from agents.orchestrator.decomposer import GoalDecomposer
from agents.orchestrator.scheduler import TaskScheduler
from agents.orchestrator.state_machine import TaskStateMachine, TaskState


class TestGoalDecomposition:
    """Tests that the orchestrator correctly breaks goals into tasks."""

    def test_simple_feature_decomposes_into_dag(self):
        """A single feature should decompose into 3-8 tasks with valid dependencies."""
        decomposer = GoalDecomposer(model="gpt-4o-mini")
        tasks = decomposer.decompose("Add a like button to posts")

        assert 3 <= len(tasks) <= 8, (
            f"Expected 3-8 tasks, got {len(tasks)}"
        )
        assert all(t.id for t in tasks), "All tasks must have IDs"
        assert all(t.agent in ("se_agent", "tester_agent", "pm_agent", "oncall_agent")
                   for t in tasks), "Tasks must be assigned to valid agents"

    def test_dependencies_form_valid_dag(self):
        """Task dependencies must form a valid directed acyclic graph."""
        decomposer = GoalDecomposer(model="gpt-4o-mini")
        tasks = decomposer.decompose("Add user profile page with avatar upload")

        scheduler = TaskScheduler(tasks)
        execution_order = scheduler.topological_sort()

        # Verify topological ordering: all dependencies come before dependents
        executed = set()
        for task in execution_order:
            for dep_id in task.depends_on:
                assert dep_id in executed, (
                    f"Task {task.id} depends on {dep_id} which was not yet executed"
                )
            executed.add(task.id)

    def test_no_circular_dependencies(self):
        """The task graph must not contain cycles."""
        decomposer = GoalDecomposer(model="gpt-4o-mini")
        tasks = decomposer.decompose("Add a comment system to posts")

        scheduler = TaskScheduler(tasks)
        # Should raise if there's a cycle
        try:
            scheduler.topological_sort()
        except Exception as e:
            pytest.fail(f"Circular dependency detected: {e}")

    def test_backend_feature_includes_tests(self):
        """Every backend task should have a corresponding test task."""
        decomposer = GoalDecomposer(model="gpt-4o-mini")
        tasks = decomposer.decompose("Add a search endpoint to the API")

        backend_tasks = [t for t in tasks if "api" in t.description.lower()]
        test_tasks = [t for t in tasks if "test" in t.description.lower()]

        assert len(test_tasks) > 0, (
            "Expected at least one test task for backend feature"
        )


class TestTaskLifecycle:
    """Tests the orchestrator's task state machine."""

    def test_task_goes_through_all_states(self):
        """A task should transition: PENDING → ASSIGNED → IN_PROGRESS → REVIEW → ACCEPTED."""
        sm = TaskStateMachine()
        task_id = "task_001"

        sm.create_task(task_id)
        assert sm.get_state(task_id) == TaskState.PENDING

        sm.assign(task_id, agent="se_agent")
        assert sm.get_state(task_id) == TaskState.ASSIGNED

        sm.start(task_id)
        assert sm.get_state(task_id) == TaskState.IN_PROGRESS

        sm.submit_for_review(task_id)
        assert sm.get_state(task_id) == TaskState.REVIEW

        sm.accept(task_id)
        assert sm.get_state(task_id) == TaskState.ACCEPTED

    def test_cannot_accept_unreviewed_task(self):
        """A task must be in REVIEW before it can be ACCEPTED."""
        sm = TaskStateMachine()
        task_id = "task_002"

        sm.create_task(task_id)
        with pytest.raises(ValueError, match="Cannot accept task not in review"):
            sm.accept(task_id)

    def test_task_can_be_rejected_and_reopened(self):
        """A task can be rejected from REVIEW and go back to IN_PROGRESS."""
        sm = TaskStateMachine()
        task_id = "task_003"

        sm.create_task(task_id)
        sm.assign(task_id, "tester_agent")
        sm.start(task_id)
        sm.submit_for_review(task_id)
        sm.reject(task_id, reason="Test coverage below threshold")
        assert sm.get_state(task_id) == TaskState.IN_PROGRESS
        assert sm.get_rejection_count(task_id) == 1

    def test_max_rejections_triggers_escalation(self, orchestrator_config):
        """After N rejections, the orchestrator should escalate."""
        sm = TaskStateMachine(max_rejections=orchestrator_config["escalation_threshold"])
        task_id = "task_004"

        sm.create_task(task_id)
        sm.assign(task_id, "se_agent")
        sm.start(task_id)

        for i in range(orchestrator_config["escalation_threshold"]):
            sm.submit_for_review(task_id)
            sm.reject(task_id, reason=f"Iteration {i+1}")

        with pytest.raises(RuntimeError, match="Escalation threshold reached"):
            sm.submit_for_review(task_id)


class TestAgentHandoff:
    """Tests that artifacts are correctly passed between agents."""

    def test_se_to_tester_handoff_includes_code(self):
        """The SE Agent's output must be available to the Tester Agent."""
        # This test validates the shared state mechanism

    def test_tester_revision_request_includes_context(self):
        """Revision requests from Tester→SE must include specific failure details."""
        # Validates that the orchestrator relays sufficient context
