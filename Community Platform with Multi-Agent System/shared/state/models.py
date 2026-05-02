"""
Shared state data models for the multi-agent system.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskGraph:
    """Represents the task dependency graph at a point in time."""
    tasks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    """Serializable state of a session for checkpointing."""
    session_id: str
    goal: str
    timestamp: str
    task_graph: TaskGraph
    iteration_count: int
    total_cost_usd: float


@dataclass
class Checkpoint:
    """A single checkpoint with metadata."""
    id: str
    session_id: str
    state: SessionState
    created_at: str


@dataclass
class AgentContext:
    """Context maintained for each agent within a session."""
    agent_id: str
    session_id: str
    current_task_id: str | None = None
    completed_task_ids: list[str] = field(default_factory=list)
    token_usage: int = 0
    heartbeat_count: int = 0
    last_heartbeat: str | None = None
