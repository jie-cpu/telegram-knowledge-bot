"""
Software Engineering Agent — generates real backend code files via LLM.

When the orchestrator assigns a coding task, this module:
1. Reads the existing project context (existing files, DB schema, API structure)
2. Calls an LLM with structured output to generate code
3. Writes the generated files to the platform/backend/ directory
4. Returns a summary of what was created
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from shared.llm_client import LLMClient, APIKeyMissingError


# ── Prompts ────────────────────────────────────────────────────────────────

SE_AGENT_SYSTEM_PROMPT = """You are a senior backend developer building a FastAPI application.
Write production-quality Python code with these standards:

1. **All endpoints must be async** — use `async def` throughout
2. **Use Pydantic v2 models** for all request/response schemas with proper validation
3. **Proper error handling** — use HTTPException with appropriate status codes (400, 401, 403, 404, 422, 500)
4. **Type hints** on every function signature and return type
5. **Docstrings** on every module, class, and public function (Google style)
6. **Dependency injection** via FastAPI `Depends()` for auth, db sessions, etc.
7. **Input validation** — never trust user input, always validate types, lengths, and formats
8. **Security** — parameterized queries (never raw SQL concatenation), proper auth checks
9. **Naming conventions** — snake_case for functions/variables, PascalCase for classes/models
10. **Separation of concerns** — routes in api/, business logic in services/, models in models/

Common patterns to follow:
- Routes accept `db: AsyncSession = Depends(get_db)` for database access
- Routes accept `current_user: User = Depends(get_current_user)` for auth
- Response models use `from_attributes=True` config for ORM mode
- File uploads validate content_type and file size before processing"""

SE_AGENT_USER_PROMPT = """You are implementing a feature for an existing FastAPI application.

## Feature Description
{feature_description}

## Existing Project Structure
{project_structure}

## Existing Database Models
{database_schema}

## Existing API Endpoints
{existing_endpoints}

## Task Requirements
{task_requirements}

## File Path Rules
- Backend Python: prefix with "app/api/", "app/models/", "app/services/"
- Frontend code: prefix with "frontend/src/components/", "frontend/src/services/", "frontend/src/types/"
- Test files: prefix with "app/tests/" (backend) or "frontend/tests/" (frontend)

## Output Format
Respond with ONLY a JSON object:
{{
  "files": {{
    "app/api/profiles.py": "code...",
    "app/models/profile.py": "code...",
    "frontend/src/components/ProfilePage.tsx": "code..."
  }},
  "summary": {{
    "new_files_created": ["app/api/profiles.py"],
    "existing_files_modified": [],
    "dependencies_added": [],
    "migration_notes": "Run: alembic revision --autogenerate"
  }}
}}

## IMPORTANT
- Match the EXACT task: if asked to write TESTS, ONLY write test files
- Only generate code that does NOT already exist (check existing structure above)
- Every file needs a module docstring or header comment
- Backend: async/await, type hints, HTTPException for errors
- Frontend: TypeScript, proper React patterns
- Never generate files unrelated to the task"""


class ContextReader:
    """Reads the existing project to build context for code generation."""

    def __init__(self, backend_path: str = "platform/backend"):
        self.backend_path = Path(backend_path)
        self.frontend_path = Path("platform/frontend")

    def read_project_structure(self) -> str:
        """Build a tree of existing files in the project."""
        lines = []

        if self.backend_path.exists():
            lines.append("  [backend]")
            for path in sorted(self.backend_path.rglob("*")):
                if path.is_file() and path.suffix in (".py", ".txt") and "__pycache__" not in str(path):
                    rel = path.relative_to(self.backend_path)
                    lines.append(f"    {rel}")

        if self.frontend_path.exists():
            lines.append("  [frontend]")
            for path in sorted(self.frontend_path.rglob("*")):
                if path.is_file() and "__pycache__" not in str(path) and "node_modules" not in str(path):
                    rel = path.relative_to(self.frontend_path)
                    lines.append(f"    {rel}")

        return "\n".join(lines) if lines else "(empty project — no existing files)"

    def read_existing_models(self) -> str:
        """Read existing model files to understand current schema."""
        models_dir = self.backend_path / "app" / "models"
        if not models_dir.exists():
            return "(no models directory yet)"

        content = ""
        for path in sorted(models_dir.glob("*.py")):
            if path.name != "__init__.py":
                content += f"\n# --- {path.name} ---\n"
                content += path.read_text() + "\n"
        return content or "(no model files yet)"

    def read_existing_routes(self) -> str:
        """Read existing API route files."""
        api_dir = self.backend_path / "app" / "api"
        if not api_dir.exists():
            return "(no API directory yet)"

        content = ""
        for path in sorted(api_dir.glob("*.py")):
            if path.name != "__init__.py":
                content += f"\n# --- {path.name} ---\n"
                for line in path.read_text().split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("@router.") or stripped.startswith("async def") or stripped.startswith("def "):
                        content += f"  {stripped}\n"
                content += "\n"
        return content or "(no route files yet)"

    def read_existing_dependency_injection(self) -> str:
        """Read the main app file to understand DI patterns."""
        main_path = self.backend_path / "app" / "__init__.py"
        main_alt = self.backend_path / "app" / "main.py"
        path = main_path if main_path.exists() else main_alt

        if path.exists():
            return path.read_text()
        return "(no main app file yet)"

    def read_frontend_structure(self) -> str:
        """Read existing frontend files for context."""
        if not self.frontend_path.exists():
            return "(no frontend directory yet)"
        files = sorted(self.frontend_path.rglob("*"))
        lines = []
        for path in files:
            if path.is_file() and "node_modules" not in str(path):
                rel = path.relative_to(self.frontend_path)
                lines.append(f"  {rel}")
        return "\n".join(lines) if lines else "(empty frontend)"


class CodeWriter:
    """Writes generated code files to disk safely.

    Routes files to correct directories:
    - app/...  → platform/backend/app/...
    - frontend/... → platform/frontend/...
    - (other)  → platform/backend/(other)
    """

    def __init__(self, backend_path: str = "platform/backend"):
        self.backend_path = Path(backend_path)
        self.frontend_path = Path("platform/frontend")
        self.project_root = Path(".")

    def write(self, file_path: str, content: str) -> Path:
        """Write a single file, routing to correct directory.

        Args:
            file_path: Relative path like "app/api/profiles.py" or
                       "frontend/src/components/ProfilePage.tsx"
            content: File contents to write

        Returns:
            Absolute path to the written file
        """
        # Route file to correct base directory
        if file_path.startswith("frontend/") or file_path.startswith("frontend\\"):
            # Strip 'frontend/' prefix — file_path is relative to frontend root
            rel_path = file_path[len("frontend/"):]
            abs_path = (self.frontend_path / rel_path).resolve()
        else:
            # Backend or shared file
            abs_path = (self.backend_path / file_path).resolve()

        # Safety: ensure we're writing within allowed directories
        allowed = [
            str(self.backend_path.resolve()),
            str(self.frontend_path.resolve()),
        ]
        if not any(str(abs_path).startswith(a) for a in allowed):
            raise ValueError(
                f"Security violation: attempted to write outside project: {abs_path}"
            )

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
        return abs_path

    def write_many(self, files: dict[str, str]) -> list[Path]:
        """Write multiple files.

        Args:
            files: Dict mapping relative paths to file contents

        Returns:
            List of absolute paths written
        """
        paths = []
        for file_path, content in files.items():
            path = self.write(file_path, content)
            paths.append(path)
        return paths


class SECoder:
    """Software Engineering Agent — generates and writes code.

    Usage:
        coder = SECoder()
        result = coder.implement(
            feature_description="Add a user profile endpoint",
            task_requirements="Create GET/PUT /api/v1/profiles/me",
        )
        # result contains the list of written files and summary
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        llm_client: LLMClient | None = None,
        backend_path: str = "platform/backend",
        sandbox: bool = False,
    ):
        self.config_path = config_path
        self.backend_path = backend_path
        self.sandbox = sandbox
        self.context = ContextReader(backend_path)
        self.writer = CodeWriter(backend_path)
        self._llm = llm_client
        self._llm_available = False

        if self._llm is None:
            try:
                self._llm = LLMClient.from_config(config_path)
                self._llm_available = True
            except (FileNotFoundError, APIKeyMissingError):
                self._llm_available = False

    def implement(
        self,
        feature_description: str,
        task_requirements: str | None = None,
        task_acceptance_criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """Main entry point: generate code and write it to disk.

        Args:
            feature_description: What feature to implement (from the PM Agent's PRD).
            task_requirements: Specific requirements for this coding task.
            task_acceptance_criteria: What "done" looks like.

        Returns:
            Dict with:
                - files_written: list of relative paths
                - files_modified: list of relative paths
                - summary: text summary of what was done
                - token_usage: estimated tokens used
                - error: error message if something failed
        """
        result = {
            "files_written": [],
            "files_modified": [],
            "summary": "",
            "token_usage": 0,
            "cost_usd": 0.0,
            "error": None,
        }

        if not self._llm_available or not self._llm:
            result["error"] = "No LLM configured — cannot generate code"
            result["summary"] = "Skipped: no LLM client available"
            return result

        # Build context
        project_structure = self.context.read_project_structure()
        database_schema = self.context.read_existing_models()
        existing_endpoints = self.context.read_existing_routes()

        reqs = task_requirements or f"Implement: {feature_description}"
        if task_acceptance_criteria:
            reqs += "\n\nAcceptance Criteria:\n" + "\n".join(
                f"- {c}" for c in task_acceptance_criteria
            )

        user_prompt = SE_AGENT_USER_PROMPT.format(
            feature_description=feature_description,
            project_structure=project_structure,
            database_schema=database_schema,
            existing_endpoints=existing_endpoints,
            task_requirements=reqs,
        )

        output_schema = {
            "files": {"filename.py": "file contents as string"},
            "summary": {
                "new_files_created": ["path/to/file.py"],
                "existing_files_modified": [],
                "dependencies_added": [],
                "migration_notes": "Any migration steps needed",
            },
        }

        print(f"  [se_agent] Calling LLM to generate code...")
        try:
            llm_result = self._llm.structured_chat(
                system_prompt=SE_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as e:
            result["error"] = f"LLM code generation failed: {e}"
            result["summary"] = str(e)
            return result

        # Record token usage and cost
        result["token_usage"] = self._llm.total_tokens if self._llm else 0
        result["cost_usd"] = self._llm.last_call_cost_usd if self._llm else 0.0

        files = llm_result.get("files", {})
        summary = llm_result.get("summary", {})

        if not files:
            result["error"] = "LLM returned no files"
            result["summary"] = "No code generated"
            return result

        # Write files to disk
        new_files = summary.get("new_files_created", [])
        modified_files = summary.get("existing_files_modified", [])

        for file_path, content in files.items():
            dest = self.writer.write(file_path, content)
            if file_path in new_files:
                result["files_written"].append(str(dest))
            elif file_path in modified_files:
                result["files_modified"].append(str(dest))
            else:
                # Infer: if file already existed, it's a modification
                # This is a simplification — in production we'd diff
                result["files_written"].append(str(dest))

        result["summary"] = (
            f"Generated {len(files)} file(s): "
            f"{len(new_files)} new, {len(modified_files)} modified"
        )
        if summary.get("migration_notes"):
            result["summary"] += f" | Note: {summary['migration_notes']}"

        print(f"  [se_agent] Files written:")
        for f in result["files_written"]:
            print(f"    + {f}")
        for f in result["files_modified"]:
            print(f"    ~ {f}")

        return result
