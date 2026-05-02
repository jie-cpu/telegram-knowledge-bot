"""File manager tool — reads/writes/list files in a sandboxed directory.

Demonstrates:
- Working with the filesystem
- Path traversal protection
- Rate limiting pattern
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.tools.base import BaseTool


class FileManagerTool(BaseTool):
    """Read, write, and list files in a sandboxed working directory."""

    def __init__(self, sandbox_dir: str | None = None) -> None:
        super().__init__()
        self._sandbox_dir = Path(sandbox_dir or "/tmp/toolagent_sandbox/").resolve()
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return (
            "Read, write, list, and delete files in a sandboxed directory. "
            "Use this to save or retrieve data from files. "
            "Operations: read, write, list, delete, info. "
            "All paths are relative to the sandbox directory. Path traversal is blocked."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete", "info"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path within the sandbox directory",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' operation)",
                },
            },
            "required": ["operation", "path"],
        }

    async def _run(
        self,
        operation: str,
        path: str,
        content: str | None = None,
    ) -> dict[str, Any]:
        # ── Path traversal protection ───────────────────────────────
        safe_path = self._resolve_path(path)

        if operation == "list":
            return self._list_files(safe_path, path)
        elif operation == "read":
            return self._read_file(safe_path, path)
        elif operation == "write":
            return self._write_file(safe_path, path, content)
        elif operation == "delete":
            return self._delete_file(safe_path, path)
        elif operation == "info":
            return self._file_info(safe_path, path)
        else:
            return {"error": f"Unknown operation: {operation}"}

    def _resolve_path(self, path: str) -> Path:
        """Resolve path and prevent traversal outside sandbox."""
        # Handle root path "/" or empty path — point to sandbox directory
        clean_path = path.strip().lstrip("/") if path.strip() not in ("", "/") else "."
        resolved = (self._sandbox_dir / clean_path).resolve()
        # Ensure resolved path is within sandbox
        resolved_str = str(resolved)
        sandbox_str = str(self._sandbox_dir)
        if not resolved_str.startswith(sandbox_str):
            raise ValueError(f"Path traversal detected: '{path}' is outside the sandbox directory")
        return resolved

    def _list_files(self, safe_path: Path, original_path: str) -> dict[str, Any]:
        target = safe_path if safe_path.exists() else self._sandbox_dir
        if not target.is_dir():
            return {"error": f"Not a directory: '{original_path}'"}

        files = []
        for entry in sorted(target.iterdir()):
            files.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                "modified": entry.stat().st_mtime,
            })

        return {
            "path": str(target.relative_to(self._sandbox_dir)) if target != self._sandbox_dir else "/",
            "file_count": len(files),
            "files": files,
        }

    def _read_file(self, safe_path: Path, original_path: str) -> dict[str, Any]:
        if not safe_path.exists():
            return {"error": f"File not found: '{original_path}'"}
        if not safe_path.is_file():
            return {"error": f"Not a file: '{original_path}'"}

        content = safe_path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": original_path,
            "size_bytes": safe_path.stat().st_size,
            "content": content,
            "line_count": content.count("\n") + 1,
        }

    def _write_file(self, safe_path: Path, original_path: str, content: str | None) -> dict[str, Any]:
        if content is None:
            return {"error": "Content is required for 'write' operation"}

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return {
            "path": original_path,
            "size_bytes": safe_path.stat().st_size,
            "message": f"File written successfully ({len(content)} chars)",
        }

    def _delete_file(self, safe_path: Path, original_path: str) -> dict[str, Any]:
        if not safe_path.exists():
            return {"error": f"File not found: '{original_path}'"}
        safe_path.unlink()
        return {"path": original_path, "message": "File deleted successfully"}

    def _file_info(self, safe_path: Path, original_path: str) -> dict[str, Any]:
        if not safe_path.exists():
            return {"error": f"File not found: '{original_path}'"}

        stat = safe_path.stat()
        return {
            "path": original_path,
            "type": "directory" if safe_path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
        }
