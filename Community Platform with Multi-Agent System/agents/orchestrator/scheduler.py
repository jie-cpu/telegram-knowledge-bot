"""
Task scheduler — dependency graph management and topological ordering.
"""

from agents.orchestrator.decomposer import Task


class CyclicDependencyError(ValueError):
    """Raised when the task graph contains a cycle."""


class TaskScheduler:
    """Manages task dependencies and produces an execution order."""

    def __init__(self, tasks: list[Task]):
        self.tasks = {t.id: t for t in tasks}

    def topological_sort(self) -> list[Task]:
        """Return tasks in dependency order (dependencies before dependents).

        Uses Kahn's algorithm. Raises CyclicDependencyError if a cycle exists.
        """
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}

        for task_id in self.tasks:
            in_degree[task_id] = 0
            adjacency[task_id] = []

        for task_id, task in self.tasks.items():
            for dep_id in task.depends_on:
                if dep_id not in self.tasks:
                    raise ValueError(
                        f"Task {task_id} depends on unknown task {dep_id}"
                    )
                adjacency[dep_id].append(task_id)
                in_degree[task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            tid = queue.pop(0)
            result.append(self.tasks[tid])
            for neighbor in adjacency.get(tid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.tasks):
            missing = set(self.tasks.keys()) - {t.id for t in result}
            raise CyclicDependencyError(
                f"Cycle detected involving tasks: {missing}"
            )

        return result
