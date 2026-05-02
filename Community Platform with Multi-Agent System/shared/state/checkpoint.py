"""
Checkpoint manager — serializes and restores session state for long-running (16h+) resilience.
"""

import json
import os
import glob
from datetime import datetime

from shared.state.models import SessionState, TaskGraph, Checkpoint


class CheckpointManager:
    """Manages disk-persisted checkpoints for long-running agent sessions.

    Features:
    - Rolling window retention (keeps N most recent)
    - Crash recovery: any checkpoint can be loaded by ID or "latest"
    - File-per-checkpoint storage for durability
    """

    def __init__(self, storage_path: str = "data/checkpoints", max_checkpoints: int = 50):
        self.storage_path = storage_path
        self.max_checkpoints = max_checkpoints
        os.makedirs(storage_path, exist_ok=True)

    def save(self, session: SessionState) -> str:
        """Save a checkpoint and return its ID."""
        checkpoint_id = (
            f"cp_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_id=session.session_id,
            state=session,
            created_at=datetime.now().isoformat(),
        )

        filepath = os.path.join(self.storage_path, f"{checkpoint_id}.json")
        with open(filepath, "w") as f:
            json.dump({
                "id": checkpoint.id,
                "session_id": checkpoint.session_id,
                "created_at": checkpoint.created_at,
                "state": {
                    "session_id": session.session_id,
                    "goal": session.goal,
                    "timestamp": session.timestamp,
                    "task_graph": {"tasks": session.task_graph.tasks},
                    "iteration_count": session.iteration_count,
                    "total_cost_usd": session.total_cost_usd,
                },
            }, f, indent=2)

        self._enforce_rolling_window(session.session_id)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> SessionState:
        """Load a checkpoint by ID."""
        filepath = os.path.join(self.storage_path, f"{checkpoint_id}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(filepath) as f:
            data = json.load(f)

        state_data = data["state"]
        return SessionState(
            session_id=state_data["session_id"],
            goal=state_data["goal"],
            timestamp=state_data["timestamp"],
            task_graph=TaskGraph(tasks=state_data["task_graph"]["tasks"]),
            iteration_count=state_data["iteration_count"],
            total_cost_usd=state_data["total_cost_usd"],
        )

    def get_latest(self, session_id: str) -> SessionState | None:
        """Get the most recent checkpoint for a session."""
        pattern = os.path.join(self.storage_path, f"cp_{session_id}_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            return None
        cp_id = os.path.splitext(os.path.basename(files[0]))[0]
        return self.load(cp_id)

    def list(self, session_id: str | None = None) -> list[str]:
        """List all checkpoint IDs, optionally filtered by session."""
        if session_id:
            pattern = os.path.join(self.storage_path, f"cp_{session_id}_*.json")
        else:
            pattern = os.path.join(self.storage_path, "cp_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def _enforce_rolling_window(self, session_id: str) -> None:
        """Remove oldest checkpoints beyond max_checkpoints."""
        pattern = os.path.join(self.storage_path, f"cp_{session_id}_*.json")
        files = sorted(glob.glob(pattern))
        while len(files) > self.max_checkpoints:
            os.remove(files.pop(0))
