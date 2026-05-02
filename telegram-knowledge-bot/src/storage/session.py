"""Session persistence via SQLite.

Stores conversation state, processing history, and user preferences so the
bot survives restarts gracefully.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id INTEGER NOT NULL,
    session_key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, session_key)
);

CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    message_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    article_id TEXT,
    tokens_used INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS articles_index (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    categories TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    file_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SessionStore:
    """Thin SQLite wrapper for session persistence."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("Session store ready at %s", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ---- session key/value ----

    def get(self, user_id: int, key: str, default: Any = None) -> Any:
        assert self._conn
        row = self._conn.execute(
            "SELECT value FROM sessions WHERE user_id = ? AND session_key = ?",
            (user_id, key),
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    def set(self, user_id: int, key: str, value: Any) -> None:
        assert self._conn
        serialized = json.dumps(value) if not isinstance(value, str) else value
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions (user_id, session_key, value, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (user_id, key, serialized),
        )
        self._conn.commit()

    def delete(self, user_id: int, key: str) -> None:
        assert self._conn
        self._conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND session_key = ?",
            (user_id, key),
        )
        self._conn.commit()

    def get_all_preferences(self, user_id: int) -> dict[str, Any]:
        assert self._conn
        rows = self._conn.execute(
            "SELECT session_key, value FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        result: dict[str, Any] = {}
        for key, value in rows:
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value
        return result

    # ---- processing log ----

    def log_processing(
        self,
        message_id: str,
        user_id: int,
        message_type: str,
        status: str = "queued",
    ) -> int:
        assert self._conn
        cur = self._conn.execute(
            """INSERT INTO processing_log (message_id, user_id, message_type, status)
               VALUES (?, ?, ?, ?)""",
            (message_id, user_id, message_type, status),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_processing(
        self,
        log_id: int,
        status: str,
        article_id: str | None = None,
        tokens_used: int = 0,
        error: str | None = None,
    ) -> None:
        assert self._conn
        self._conn.execute(
            """UPDATE processing_log
               SET status = ?,
                   article_id = ?,
                   tokens_used = ?,
                   error = ?,
                   completed_at = CASE WHEN ? IN ('completed','failed') THEN datetime('now') ELSE NULL END
               WHERE id = ?""",
            (status, article_id, tokens_used, error, status, log_id),
        )
        self._conn.commit()

    # ---- article index ----

    def index_article(self, article_id: str, title: str, summary: str,
                      categories: list[str], tags: list[str],
                      file_path: str, source_type: str, user_id: int) -> None:
        assert self._conn
        self._conn.execute(
            """INSERT OR REPLACE INTO articles_index
               (article_id, title, summary, categories, tags, file_path, source_type, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (article_id, title, summary,
             json.dumps(categories), json.dumps(tags),
             file_path, source_type, user_id),
        )
        self._conn.commit()

    def search_articles(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        assert self._conn
        like = f"%{query}%"
        rows = self._conn.execute(
            """SELECT article_id, title, summary, categories, tags, file_path, source_type, created_at
               FROM articles_index
               WHERE title LIKE ? OR summary LIKE ? OR categories LIKE ? OR tags LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (like, like, like, like, limit),
        ).fetchall()
        return [
            {
                "article_id": r[0],
                "title": r[1],
                "summary": r[2],
                "categories": json.loads(r[3]),
                "tags": json.loads(r[4]),
                "file_path": r[5],
                "source_type": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    def recent_articles(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        assert self._conn
        rows = self._conn.execute(
            """SELECT article_id, title, summary, categories, tags, file_path, source_type, created_at
               FROM articles_index
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "article_id": r[0],
                "title": r[1],
                "summary": r[2],
                "categories": json.loads(r[3]),
                "tags": json.loads(r[4]),
                "file_path": r[5],
                "source_type": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]
