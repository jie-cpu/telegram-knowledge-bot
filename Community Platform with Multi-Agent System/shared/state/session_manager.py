"""
Session manager — checkpoint/resume, graceful shutdown, heartbeat monitoring.

This is the core of the "Long-running agent management (16+ hours)" capability.
It ensures that the orchestrator can survive process restarts, crashes,
and maintain full execution state across extended sessions.
"""

import os
import json
import signal
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from shared.state.models import SessionState, TaskGraph, AgentContext
from shared.state.checkpoint import CheckpointManager


class GracefulShutdown(Exception):
    """Raised when SIGINT/SIGTERM is received, triggering checkpoint save."""


class SessionManager:
    """Manages the lifecycle of a long-running orchestrator session.

    Features:
    - Creates and tracks sessions with unique IDs
    - Auto-checkpoints after each task (configurable interval)
    - Graceful shutdown on Ctrl+C (saves state before exit)
    - Resumes from any checkpoint
    - Tracks agent heartbeats and detects failures
    - Reports session duration and resource usage
    """

    def __init__(
        self,
        session_id: str | None = None,
        storage_path: str = "data/checkpoints",
        checkpoint_interval: int = 300,
        max_checkpoints: int = 50,
    ):
        self.storage_path = Path(storage_path)
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_time = time.time()

        self.checkpoint_mgr = CheckpointManager(
            storage_path=str(self.storage_path),
            max_checkpoints=max_checkpoints,
        )

        # Session state
        self.session_id = session_id or self._generate_session_id()
        self.goal: str = ""
        self.tasks: list[dict] = []
        self.task_states: dict[str, str] = {}
        self.task_results: dict[str, Any] = {}
        self.iteration_count: int = 0
        self.total_cost_usd: float = 0.0
        self.start_time: float = time.time()
        self.last_activity: str = datetime.now(timezone.utc).isoformat()

        # Agent heartbeat tracking
        self.agent_heartbeats: dict[str, dict] = {
            "pm_agent": {"last_seen": None, "missed_count": 0, "status": "unknown"},
            "se_agent": {"last_seen": None, "missed_count": 0, "status": "unknown"},
            "tester_agent": {"last_seen": None, "missed_count": 0, "status": "unknown"},
            "oncall_agent": {"last_seen": None, "missed_count": 0, "status": "unknown"},
        }
        self.max_missed_heartbeats: int = 3
        self.heartbeat_interval: int = 60

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

    # ── Session Lifecycle ──────────────────────────────────────────────

    def create(self, goal: str, tasks: list) -> str:
        """Create a new session with the given goal and task list."""
        self.goal = goal
        self.tasks = [self._task_to_dict(t) for t in tasks]
        self.task_states = {t.id if hasattr(t, 'id') else t.get('id', ''): 'pending'
                           for t in tasks}
        self.start_time = time.time()

        print(f"  [session] Created session: {self.session_id}")
        return self.session_id

    def save_checkpoint(self, reason: str = "periodic") -> str | None:
        """Save current state as a checkpoint.

        Args:
            reason: Why the checkpoint is being saved (periodic, task_complete,
                   shutdown, etc.)

        Returns:
            Checkpoint ID if saved, None if interval hasn't elapsed
            (for periodic checkpoints).
        """
        now = time.time()

        # For periodic checkpoints, respect the interval
        if reason == "periodic":
            if now - self.last_checkpoint_time < self.checkpoint_interval:
                return None

        session_state = SessionState(
            session_id=self.session_id,
            goal=self.goal,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_graph=TaskGraph(tasks=[
                {
                    "id": tid,
                    "status": self.task_states.get(tid, "unknown"),
                    "agent": self._get_task_agent(tid),
                    "description": self._get_task_description(tid),
                }
                for tid in self.task_states
            ]),
            iteration_count=self.iteration_count,
            total_cost_usd=self.total_cost_usd,
        )

        cp_id = self.checkpoint_mgr.save(session_state)
        self.last_checkpoint_time = now

        print(f"  [session] Checkpoint saved: {cp_id} ({reason})")
        return cp_id

    def resume(self, session_id: str) -> bool:
        """Resume a session from its latest checkpoint.

        Args:
            session_id: Session to resume.

        Returns:
            True if resumed successfully, False if no checkpoint found.
        """
        state = self.checkpoint_mgr.get_latest(session_id)
        if state is None:
            print(f"  [session] No checkpoint found for session: {session_id}")
            return False

        self.session_id = state.session_id
        self.goal = state.goal
        self.iteration_count = state.iteration_count
        self.total_cost_usd = state.total_cost_usd

        # Restore task states from the task graph
        for task_data in state.task_graph.tasks:
            tid = task_data.get("id")
            self.task_states[tid] = task_data.get("status", "unknown")

        print(f"  [session] Resumed session {session_id}")
        print(f"  [session]   Goal: {self.goal}")
        print(f"  [session]   Tasks: {len(state.task_graph.tasks)} "
              f"({sum(1 for t in state.task_graph.tasks if t.get('status') == 'completed')} completed)")
        print(f"  [session]   Iterations: {self.iteration_count}")
        print(f"  [session]   Previous cost: ${self.total_cost_usd:.4f}")

        return True

    def close(self, reason: str = "completed"):
        """Close the session, saving a final checkpoint."""
        self.save_checkpoint(reason=f"session_{reason}")
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_duration(elapsed)
        print(f"  [session] Session {self.session_id} closed ({reason})")
        print(f"  [session]   Duration: {elapsed_str}")
        print(f"  [session]   Checkpoints: {len(self.checkpoint_mgr.list(self.session_id))}")
        print(f"  [session]   Est. cost: ${self.total_cost_usd:.4f}")

    # ── Heartbeat Monitoring ───────────────────────────────────────────

    def record_heartbeat(self, agent_id: str):
        """Record a heartbeat from an agent.

        Called by the orchestrator when an agent reports in.
        """
        if agent_id not in self.agent_heartbeats:
            self.agent_heartbeats[agent_id] = {
                "last_seen": None, "missed_count": 0, "status": "unknown"
            }

        heartbeat = self.agent_heartbeats[agent_id]
        now = datetime.now(timezone.utc).isoformat()

        # Calculate missed count based on elapsed time
        if heartbeat["last_seen"] is not None:
            last = datetime.fromisoformat(heartbeat["last_seen"])
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            expected_beats = int(elapsed / self.heartbeat_interval)
            heartbeat["missed_count"] = max(0, expected_beats - 1)

        heartbeat["last_seen"] = now
        heartbeat["status"] = "alive"

    def check_agent_health(self) -> dict[str, str]:
        """Check all agents and return health status.

        Returns:
            Dict mapping agent_id → "alive", "missed_beats", or "dead"
        """
        statuses = {}
        now = datetime.now(timezone.utc)

        for agent_id, hb in self.agent_heartbeats.items():
            if hb["last_seen"] is None:
                statuses[agent_id] = "never_connected"
                continue

            last = datetime.fromisoformat(hb["last_seen"])
            elapsed = (now - last).total_seconds()
            missed = int(elapsed / self.heartbeat_interval)

            if missed >= self.max_missed_heartbeats:
                statuses[agent_id] = "dead"
                hb["status"] = "dead"
            elif missed > 0:
                statuses[agent_id] = "missed_beats"
                hb["status"] = "missed_beats"
            else:
                statuses[agent_id] = "alive"
                hb["status"] = "alive"

            hb["missed_count"] = missed

        return statuses

    def print_heartbeat_report(self):
        """Print a formatted heartbeat status report."""
        statuses = self.check_agent_health()
        print(f"\n  ── Heartbeat Status [{datetime.now().strftime('%H:%M:%S')}] ──")
        for agent_id in sorted(statuses):
            status = statuses[agent_id]
            icons = {"alive": "OK", "missed_beats": "!", "dead": "ERR", "never_connected": "-"}
            icon = icons.get(status, "?")
            hb = self.agent_heartbeats.get(agent_id, {})
            missed = hb.get("missed_count", 0)
            print(f"    {icon} {agent_id:<20} {status:<15} (missed: {missed})")

    def wait_for_heartbeats(self, duration_seconds: int = 5):
        """Simulate a period of heartbeat monitoring.

        In production this would run as a background thread. Here we
        demonstrate the pattern synchronously for interview purposes.
        """
        print(f"\n  [heartbeat] Monitoring agents for {duration_seconds}s...")
        for i in range(duration_seconds):
            time.sleep(1)
            # Simulate agents sending heartbeats
            for aid in self.agent_heartbeats:
                if i == 0:  # first second: all agents alive
                    self.record_heartbeat(aid)
            if (i + 1) % 5 == 0:
                self.print_heartbeat_report()

    # ── Activity Tracking ──────────────────────────────────────────────

    def record_task_start(self, task_id: str):
        """Record that a task has started."""
        self.task_states[task_id] = "in_progress"
        self.last_activity = datetime.now(timezone.utc).isoformat()
        self.record_heartbeat("orchestrator")

    def record_task_complete(self, task_id: str, result: dict, cost: float = 0.0):
        """Record that a task has completed."""
        self.task_states[task_id] = "completed"
        self.task_results[task_id] = result
        self.total_cost_usd += cost
        self.iteration_count += 1
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def get_pending_tasks(self) -> list[str]:
        """Get IDs of tasks that are still pending."""
        return [tid for tid, state in self.task_states.items()
                if state == "pending"]

    def get_completed_tasks(self) -> list[str]:
        """Get IDs of tasks that have completed."""
        return [tid for tid, state in self.task_states.items()
                if state == "completed"]

    def get_incomplete_tasks(self) -> list[dict]:
        """Get all task data for tasks that are not yet completed.

        Used when resuming — these tasks still need to run.
        """
        return [
            t for t in self.tasks
            if self.task_states.get(t.get("id", ""), "pending") != "completed"
        ]

    # ── Utilities ──────────────────────────────────────────────────────

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        return self._format_duration(self.elapsed_seconds)

    def _generate_session_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sess_{timestamp}"

    def _format_duration(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _task_to_dict(self, task) -> dict:
        """Convert a Task object to a dict for serialization."""
        if hasattr(task, '__dataclass_fields__'):
            return {
                "id": task.id,
                "agent": task.agent,
                "description": task.description,
                "depends_on": task.depends_on,
                "status": getattr(task, 'status', 'pending'),
                "acceptance_criteria": getattr(task, 'acceptance_criteria', []),
            }
        return task if isinstance(task, dict) else {"id": str(task)}

    def _get_task_agent(self, task_id: str) -> str:
        for t in self.tasks:
            if t.get("id") == task_id:
                return t.get("agent", "unknown")
        return "unknown"

    def _get_task_description(self, task_id: str) -> str:
        for t in self.tasks:
            if t.get("id") == task_id:
                return t.get("description", "")
        return ""

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def _handler(signum, frame):
            print(f"\n  [session] Signal {signum} received. Saving checkpoint...")
            self.save_checkpoint(reason="shutdown")
            print(f"  [session] Run --resume {self.session_id} to continue.")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def list_sessions(storage_path: str = "data/checkpoints") -> list[dict]:
    """List all available sessions with summary info.

    Useful for the --list-sessions CLI flag.
    """
    mgr = CheckpointManager(storage_path=storage_path)
    all_checkpoints = mgr.list()

    # Group by session and get latest
    sessions: dict[str, dict] = {}
    for cp_id in all_checkpoints:
        try:
            state = mgr.load(cp_id)
            if state.session_id not in sessions:
                sessions[state.session_id] = {
                    "session_id": state.session_id,
                    "goal": state.goal,
                    "tasks": len(state.task_graph.tasks),
                    "completed": sum(1 for t in state.task_graph.tasks
                                     if t.get("status") == "completed"),
                    "last_updated": state.timestamp,
                }
        except Exception:
            continue

    return sorted(sessions.values(),
                  key=lambda s: s.get("last_updated", ""),
                  reverse=True)
