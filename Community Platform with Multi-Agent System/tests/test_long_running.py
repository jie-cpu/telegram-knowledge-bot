"""
Meta-tests: Long-running agent management.

Tests checkpointing, recovery, and heartbeat mechanisms
that allow the system to survive 16+ hour sessions.
"""

import tempfile
import time

import pytest
from shared.state.checkpoint import CheckpointManager
from shared.state.models import SessionState, TaskGraph


class TestCheckpointing:
    """Tests that the system can save and restore state."""

    def test_checkpoint_serialization_roundtrip(self):
        """A checkpoint must survive serialize → deserialize → re-serialize."""
        manager = CheckpointManager(storage_path=tmp_path())

        session = SessionState(
            session_id="test_session",
            goal="Add a user profile page",
            timestamp="2026-04-15T14:30:00Z",
            task_graph=TaskGraph(
                tasks=[
                    {"id": "t1", "status": "completed"},
                    {"id": "t2", "status": "in_progress"},
                ]
            ),
            iteration_count=12,
            total_cost_usd=0.42,
        )

        checkpoint_id = manager.save(session)
        restored = manager.load(checkpoint_id)

        assert restored.session_id == session.session_id
        assert restored.goal == session.goal
        assert restored.iteration_count == session.iteration_count
        assert len(restored.task_graph.tasks) == len(session.task_graph.tasks)

    def test_checkpoint_survives_process_restart(self):
        """Checkpoints stored on disk must survive a simulated crash."""
        path = tmp_path()
        manager = CheckpointManager(storage_path=path)

        original = SessionState(
            session_id="crash_test",
            goal="Add comment system",
            timestamp="2026-04-15T16:00:00Z",
            task_graph=TaskGraph(tasks=[{"id": "t1", "status": "in_progress"}]),
            iteration_count=5,
            total_cost_usd=0.18,
        )
        cpid = manager.save(original)

        # New CheckpointManager instance, same path
        manager2 = CheckpointManager(storage_path=path)
        restored = manager2.load(cpid)

        assert restored.goal == "Add comment system"
        assert restored.iteration_count == 5
        assert restored.task_graph.tasks[0]["id"] == "t1"

    def test_latest_checkpoint_is_identifiable(self):
        """The most recent checkpoint must be findable without knowing its ID."""
        manager = CheckpointManager(storage_path=tmp_path())

        manager.save(SessionState(
            session_id="s1", goal="goal1", timestamp="2026-04-15T10:00:00Z",
            task_graph=TaskGraph(tasks=[]), iteration_count=0, total_cost_usd=0,
        ))
        time.sleep(0.01)
        manager.save(SessionState(
            session_id="s2", goal="goal2", timestamp="2026-04-15T11:00:00Z",
            task_graph=TaskGraph(tasks=[]), iteration_count=1, total_cost_usd=0.01,
        ))

        latest = manager.get_latest(session_id="s2")
        assert latest.goal == "goal2"
        assert latest.iteration_count == 1

    def test_checkpoint_rolling_window(self):
        """Only the N most recent checkpoints per session should be retained."""
        manager = CheckpointManager(storage_path=tmp_path(), max_checkpoints=3)

        for i in range(5):
            manager.save(SessionState(
                session_id="same_session", goal=f"iteration_{i}",
                timestamp=f"2026-04-15T1{i}:00:00Z",
                task_graph=TaskGraph(tasks=[]),
                iteration_count=i, total_cost_usd=0,
            ))

        checkpoints = manager.list(session_id="same_session")
        assert len(checkpoints) <= 3, (
            f"Expected ≤3 checkpoints for session, got {len(checkpoints)}"
        )


def tmp_path():
    """Create a temporary directory for checkpoint storage."""
    path = tempfile.mkdtemp(prefix="checkpoint_test_")
    return path
