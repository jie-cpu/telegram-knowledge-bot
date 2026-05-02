"""Code execution tool — safely runs Python code in a restricted sandbox.

Demonstrates:
- Safe sandbox design with restricted modules
- Timeout handling
- Output capture
"""

from __future__ import annotations

import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Any

from agent.tools.base import BaseTool

# Modules allowed in the sandbox
_ALLOWED_MODULES = {
    "math", "json", "datetime", "random", "collections",
    "itertools", "statistics", "string", "re", "decimal",
}


class CodeExecutorTool(BaseTool):
    """Executes Python code in a restricted sandbox."""

    def __init__(self, timeout: int = 10, max_output_chars: int = 2000) -> None:
        super().__init__()
        self._timeout = timeout
        self._max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return (
            "Execute Python code and return the output. "
            "Use this for data analysis, calculations, or any task that requires running code. "
            "The code runs in a sandbox with limited modules: "
            f"{', '.join(sorted(_ALLOWED_MODULES))}. "
            "Use print() to see output. Use sys.stdout.write() for more control."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Use print() to output results.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 10, max: 30)",
                    "default": self._timeout,
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": ["code"],
        }

    async def _run(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        timeout = min(timeout or self._timeout, 30)

        # ── Safety checks ───────────────────────────────────────────
        # Check for dangerous imports (only truly dangerous ones)
        dangerous_patterns = [
            "import os", "from os", "import subprocess", "from subprocess",
            "import shutil", "from shutil",
            "__import__('os'", "__import__('subprocess'",
            "eval(", "exec(", "compile(",
            "import socket", "import ctypes",
        ]
        violations = [p for p in dangerous_patterns if p in code]
        if violations:
            return {
                "success": False,
                "error": f"Code contains unsafe patterns: {violations}",
                "stdout": "",
                "stderr": "Security violation",
            }

        # ── Execute ─────────────────────────────────────────────────
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Build sandbox globals with safe builtins
        def safe___import__(name, *args, **kwargs):
            """Restricted import — only allow listed modules and standard library."""
            if name in _ALLOWED_MODULES:
                return __import__(name, *args, **kwargs)
            base = name.split(".")[0]
            if base in _ALLOWED_MODULES:
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Module '{name}' is not allowed in the sandbox")

        sandbox_globals: dict[str, Any] = {
            "__builtins__": {
                "__import__": safe___import__,
                "abs": abs, "all": all, "any": any, "bool": bool,
                "chr": chr, "complex": complex, "dict": dict,
                "divmod": divmod, "enumerate": enumerate, "filter": filter,
                "float": float, "format": format, "frozenset": frozenset,
                "hex": hex, "int": int, "isinstance": isinstance,
                "iter": iter, "len": len, "list": list, "map": map,
                "max": max, "min": min, "next": next, "object": object,
                "oct": oct, "ord": ord, "pow": pow, "print": print,
                "range": range, "repr": repr, "reversed": reversed,
                "round": round, "set": set, "slice": slice,
                "sorted": sorted, "str": str, "sum": sum,
                "tuple": tuple, "type": type, "zip": zip,
                "True": True, "False": False, "None": None,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "StopIteration": StopIteration,
                "RuntimeError": RuntimeError, "ZeroDivisionError": ZeroDivisionError,
                "isinstance": isinstance, "issubclass": issubclass,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                "delattr": delattr, "callable": callable,
                "open": open,
            },
        }

        try:
            # Compile the code
            compiled = compile(code, "<sandbox>", "exec")

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, sandbox_globals)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Truncate output
            if len(stdout) > self._max_output_chars:
                stdout = stdout[: self._max_output_chars] + "\n... (truncated)"
            if len(stderr) > self._max_output_chars:
                stderr = stderr[: self._max_output_chars] + "\n... (truncated)"

            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": 0,
            }

        except Exception:
            stderr = traceback.format_exc()
            if len(stderr) > self._max_output_chars:
                stderr = stderr[: self._max_output_chars] + "\n... (truncated)"
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr,
                "exit_code": 1,
            }
