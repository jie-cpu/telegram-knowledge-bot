"""
Goal decomposer — breaks down high-level goals into structured task DAGs.

Uses LLM for intelligent decomposition, with fallback to rule-based logic
when no API key is configured.
"""

from dataclasses import dataclass, field
from pathlib import Path

from shared.llm_client import LLMClient, APIKeyMissingError


@dataclass
class Task:
    id: str
    agent: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    acceptance_criteria: list[str] = field(default_factory=list)


DECOMPOSITION_SYSTEM_PROMPT = """You are a senior technical lead breaking down product goals into
actionable engineering tasks for a 4-agent team:

- se_agent: Full-stack developer. Writes backend (FastAPI/Python) and frontend (React/TypeScript) code.
- tester_agent: QA engineer. Writes and runs tests, checks coverage, evaluates code quality.
- pm_agent: Product manager. Refines vague goals into structured acceptance criteria.
- oncall_agent: DevOps/on-call. Deploys code, monitors health, rolls back on failure.

For each goal, produce a list of 3-7 tasks. Each task must have:
- A unique ID (task_001, task_002, ...)
- An assigned agent from the list above
- A clear, specific description of what to do
- Dependencies: list of task IDs that must complete before this one
- Acceptance criteria: 1-2 specific, testable conditions

Rules:
1. The first task should be pm_agent refining the goal with acceptance criteria.
2. Backend code tasks should come before frontend tasks (API first).
3. Every code task must have a corresponding tester_agent verification task.
4. The last task should be oncall_agent deploying and monitoring.
5. Tasks must form a valid DAG — no circular dependencies.
6. Use task IDs (task_001) in depends_on, not descriptions."""


DECOMPOSITION_USER_PROMPT = """Decompose this goal into tasks:

GOAL: {goal}

{context}Respond with ONLY a JSON object:
{{
  "tasks": [
    {{
      "id": "task_001",
      "agent": "pm_agent",
      "description": "...",
      "depends_on": [],
      "acceptance_criteria": ["..."]
    }}
  ]
}}"""


class GoalDecomposer:
    """Decomposes goals into dependency-ordered task lists.

    Primary mode: LLM-driven decomposition via structured output.
    Fallback mode: Rule-based decomposition when LLM is unavailable.

    The LLM mode understands context (what already exists in the platform)
    and adjusts the task list accordingly.
    """

    VALID_AGENTS = ("se_agent", "tester_agent", "pm_agent", "oncall_agent")

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config_path: str = "config.yaml",
        llm_client: LLMClient | None = None,
    ):
        self.model = model
        self.config_path = config_path
        self._task_counter = 0

        # Try to create an LLM client
        self._llm = llm_client
        self._llm_available = False

        if self._llm is None:
            try:
                config = Path(config_path)
                if config.exists():
                    self._llm = LLMClient.from_config(str(config))
                    self._llm_available = True
            except (FileNotFoundError, APIKeyMissingError):
                self._llm_available = False

    def decompose(self, goal: str, context: dict | None = None) -> list[Task]:
        """Decompose a goal into a list of tasks.

        Args:
            goal: Natural language goal from the user.
            context: Optional dict with platform context (existing endpoints,
                     database tables, frontend routes) to guide decomposition.

        Returns:
            List of Task objects forming a valid DAG.
        """
        self._task_counter = 0

        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty")

        if self._llm_available and self._llm:
            return self._decompose_with_llm(goal, context)

        return self._decompose_rules(goal)

    def _decompose_with_llm(
        self, goal: str, context: dict | None = None
    ) -> list[Task]:
        """Use LLM for intelligent decomposition."""
        context_str = ""
        if context:
            context_str = f"""\nExisting platform context:
- Endpoints: {context.get('endpoints', ['none'])}
- Database tables: {context.get('tables', ['none'])}
- Frontend routes: {context.get('routes', ['none'])}
\n"""

        user_prompt = DECOMPOSITION_USER_PROMPT.format(
            goal=goal, context=context_str
        )

        output_schema = {
            "tasks": [
                {
                    "id": "string",
                    "agent": "se_agent | tester_agent | pm_agent | oncall_agent",
                    "description": "string (specific, actionable)",
                    "depends_on": ["string (task IDs)"],
                    "acceptance_criteria": ["string (testable condition)"],
                }
            ]
        }

        try:
            result = self._llm.structured_chat(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as e:
            print(f"  [!] LLM decomposition failed ({e}), falling back to rules")
            return self._decompose_rules(goal)

        tasks = self._parse_llm_output(result, goal)
        self._validate_task_graph(tasks)
        return tasks

    def _decompose_rules(self, goal: str) -> list[Task]:
        """Rule-based fallback decomposition."""
        goal_lower = goal.lower()

        needs_db = any(
            word in goal_lower for word in
            ["save", "store", "database", "profile", "table", "schema", "persist"]
        )
        needs_api = any(
            word in goal_lower for word in
            ["api", "endpoint", "get", "post", "put", "delete", "upload", "crud"]
        )
        needs_frontend = any(
            word in goal_lower for word in
            ["page", "component", "form", "button", "ui", "display", "view", "frontend"]
        )
        needs_upload = any(
            word in goal_lower for word in
            ["upload", "file", "image", "avatar", "photo", "attachment", "media"]
        )

        tasks = []

        tasks.append(self._make_task(
            agent="pm_agent",
            description=f"Refine goal '{goal}' into a structured PRD with acceptance criteria",
            depends_on=[],
            acceptance_criteria=["PRD is written", "Edge cases are documented"],
        ))

        if needs_db:
            tasks.append(self._make_task(
                agent="se_agent",
                description=f"Design and write database schema migration for: {goal}",
                depends_on=[tasks[-1].id],
                acceptance_criteria=["Migration runs without errors", "Schema matches PRD"],
            ))

        if needs_api:
            tasks.append(self._make_task(
                agent="se_agent",
                description=f"Implement API endpoints for: {goal}",
                depends_on=[tasks[-1].id],
                acceptance_criteria=[
                    "All endpoints return correct status codes",
                    "Input validation rejects invalid data"
                ],
            ))

        if needs_upload:
            tasks.append(self._make_task(
                agent="se_agent",
                description=f"Implement file upload and storage for: {goal}",
                depends_on=[tasks[-1].id] if tasks else [],
                acceptance_criteria=[
                    "Validates file type and size",
                    "Stores file securely"
                ],
            ))

        if not tasks[-1].agent == "se_agent" and not needs_api:
            # always have at least one SE task
            tasks.append(self._make_task(
                agent="se_agent",
                description=f"Implement feature: {goal}",
                depends_on=[tasks[-1].id],
                acceptance_criteria=["Feature works as specified in PRD"],
            ))

        tasks.append(self._make_task(
            agent="tester_agent",
            description=f"Write and run test suite, verify coverage >= 75% for: {goal}",
            depends_on=[tasks[-1].id],
            acceptance_criteria=[
                "Test coverage >= 75%",
                "All tests pass",
                "LLM-judge score >= 6.0"
            ],
        ))

        if needs_frontend:
            tasks.append(self._make_task(
                agent="se_agent",
                description=f"Build frontend components for: {goal}",
                depends_on=[tasks[-1].id],
                acceptance_criteria=[
                    "Component renders correctly",
                    "Handles loading, empty, and error states"
                ],
            ))
            tasks.append(self._make_task(
                agent="tester_agent",
                description=f"Write frontend tests for: {goal}",
                depends_on=[tasks[-1].id],
                acceptance_criteria=["Component tests pass"],
            ))

        tasks.append(self._make_task(
            agent="oncall_agent",
            description=f"Deploy to staging, run health checks, promote to production for: {goal}",
            depends_on=[tasks[-1].id],
            acceptance_criteria=[
                "Health checks pass on staging",
                "No errors in first 5 minutes of production"
            ],
        ))

        return tasks

    def _parse_llm_output(self, result: dict, goal: str) -> list[Task]:
        """Parse LLM structured output into Task objects."""
        raw_tasks = result.get("tasks", [])
        if not raw_tasks:
            raise ValueError("LLM returned no tasks")

        tasks = []
        for raw in raw_tasks:
            task_id = raw.get("id", f"task_{len(tasks)+1:03d}")
            agent = raw.get("agent", "se_agent")

            if agent not in self.VALID_AGENTS:
                print(f"  [!] Unknown agent '{agent}' in LLM output, using se_agent")
                agent = "se_agent"

            tasks.append(Task(
                id=task_id,
                agent=agent,
                description=raw.get("description", f"Work on: {goal}"),
                depends_on=raw.get("depends_on", []),
                acceptance_criteria=raw.get("acceptance_criteria", []),
            ))

        return tasks

    def _validate_task_graph(self, tasks: list[Task]):
        """Validate the task graph: no cycles, all deps exist, valid agents."""
        task_ids = {t.id for t in tasks}

        # Check all dependency IDs exist
        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"Task {task.id} depends on unknown task {dep_id}"
                    )

        # Check for cycles using DFS
        for task in tasks:
            visited = set()
            path = set()
            self._check_cycle(task.id, tasks, visited, path)

    def _check_cycle(
        self, task_id: str, tasks: list[Task],
        visited: set, path: set
    ):
        """DFS cycle detection."""
        if task_id in path:
            raise ValueError(f"Cycle detected involving task {task_id}")
        if task_id in visited:
            return

        path.add(task_id)
        visited.add(task_id)

        task = next((t for t in tasks if t.id == task_id), None)
        if task:
            for dep_id in task.depends_on:
                self._check_cycle(dep_id, tasks, visited, path)

        path.discard(task_id)

    def _make_task(
        self, agent: str, description: str,
        depends_on: list[str],
        acceptance_criteria: list[str] | None = None,
    ) -> Task:
        self._task_counter += 1
        task_id = f"task_{self._task_counter:03d}"
        return Task(
            id=task_id,
            agent=agent,
            description=description,
            depends_on=depends_on,
            acceptance_criteria=acceptance_criteria or [],
        )
