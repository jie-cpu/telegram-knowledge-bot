"""
Task state machine — tracks and enforces valid task lifecycle transitions.
"""

from enum import Enum, auto


class TaskState(Enum):
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    REVIEW = auto()
    ACCEPTED = auto()
    REJECTED = auto()


# Valid transitions: (from, to)
_VALID_TRANSITIONS = {
    (TaskState.PENDING, TaskState.ASSIGNED),
    (TaskState.ASSIGNED, TaskState.IN_PROGRESS),
    (TaskState.IN_PROGRESS, TaskState.REVIEW),
    (TaskState.REVIEW, TaskState.ACCEPTED),
    (TaskState.REVIEW, TaskState.REJECTED),
    (TaskState.REJECTED, TaskState.IN_PROGRESS),  # rework
}


class TaskStateMachine:
    """Enforces valid state transitions for tasks.

    Each task follows: PENDING → ASSIGNED → IN_PROGRESS → REVIEW → ACCEPTED.
    From REVIEW, a task can be REJECTED and return to IN_PROGRESS for rework.
    After max_rejections, the task escalates.
    """

    def __init__(self, max_rejections: int = 5):
        self._states: dict[str, TaskState] = {}
        self._rejection_counts: dict[str, int] = {}
        self._agents: dict[str, str] = {}
        self._rejection_reasons: dict[str, list[str]] = {}
        self.max_rejections = max_rejections

    def create_task(self, task_id: str) -> None:
        if task_id in self._states:
            raise ValueError(f"Task {task_id} already exists")
        self._states[task_id] = TaskState.PENDING
        self._rejection_counts[task_id] = 0
        self._rejection_reasons[task_id] = []

    def get_state(self, task_id: str) -> TaskState:
        if task_id not in self._states:
            raise ValueError(f"Task {task_id} does not exist")
        return self._states[task_id]

    def assign(self, task_id: str, agent: str) -> None:
        self._transition(task_id, TaskState.ASSIGNED)
        self._agents[task_id] = agent

    def start(self, task_id: str) -> None:
        self._transition(task_id, TaskState.IN_PROGRESS)

    def submit_for_review(self, task_id: str) -> None:
        if self._rejection_counts.get(task_id, 0) >= self.max_rejections:
            raise RuntimeError(
                f"Escalation threshold reached for task {task_id}: "
                f"{self._rejection_counts[task_id]} rejections"
            )
        self._transition(task_id, TaskState.REVIEW)

    def accept(self, task_id: str) -> None:
        if self._states[task_id] != TaskState.REVIEW:
            raise ValueError(
                f"Cannot accept task not in review "
                f"(current state: {self._states[task_id].name})"
            )
        self._transition(task_id, TaskState.ACCEPTED)

    def reject(self, task_id: str, reason: str = "") -> None:
        if self._states[task_id] != TaskState.REVIEW:
            raise ValueError(
                f"Cannot reject task not in review "
                f"(current state: {self._states[task_id].name})"
            )
        self._rejection_counts[task_id] = self._rejection_counts.get(task_id, 0) + 1
        self._rejection_reasons[task_id].append(reason)
        self._states[task_id] = TaskState.IN_PROGRESS

    def get_rejection_count(self, task_id: str) -> int:
        return self._rejection_counts.get(task_id, 0)

    def _transition(self, task_id: str, to_state: TaskState) -> None:
        if task_id not in self._states:
            raise ValueError(f"Task {task_id} does not exist")
        from_state = self._states[task_id]
        if (from_state, to_state) not in _VALID_TRANSITIONS:
            raise ValueError(
                f"Invalid transition: {from_state.name} → {to_state.name}"
            )
        self._states[task_id] = to_state
