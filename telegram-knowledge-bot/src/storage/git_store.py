"""Git-backed article storage.

Every processed article is committed as a markdown file to a local git
repository.  This gives us version history, diff tracking, and a natural
path toward publishing (push to GitHub/GitLab).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.models.schemas import Article

logger = logging.getLogger(__name__)


class GitStore:
    """Manages a local git repo that serves as the knowledge base."""

    def __init__(self, repo_path: Path, author_name: str = "PKM Bot",
                 author_email: str = "bot@knowledge.local") -> None:
        self._path = repo_path
        self._author_name = author_name
        self._author_email = author_email
        self._repo: Repo | None = None

    # ---- lifecycle ----

    def initialize(self) -> None:
        """Open an existing repo or init a new one."""
        self._path.mkdir(parents=True, exist_ok=True)

        try:
            self._repo = Repo(self._path)
            logger.info("Opened existing git repo at %s", self._path)
        except InvalidGitRepositoryError:
            self._repo = Repo.init(self._path)
            # Create initial README so we have something to commit
            readme = self._path / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Personal Knowledge Base\n\n"
                    "Automatically generated articles from the PKM Telegram bot.\n"
                )
            self._repo.index.add(["README.md"])
            self._commit("chore: initialize knowledge base")
            logger.info("Initialised new git repo at %s", self._path)

    @property
    def repo(self) -> Repo:
        assert self._repo is not None, "call initialize() first"
        return self._repo

    # ---- article operations ----

    def save_article(self, article: Article) -> Path:
        """Write article as markdown and commit to git.

        Returns the absolute file path of the saved article.
        """
        file_path = self._path / article.file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(article.to_markdown(), encoding="utf-8")

        rel_path = str(file_path.relative_to(self._path))
        self.repo.index.add([rel_path])
        self._commit(f"article: add {article.title}")
        logger.info("Saved and committed article: %s", rel_path)
        return file_path

    def read_article(self, file_path: str) -> str | None:
        """Read an article markdown file by its repository-relative path."""
        full = self._path / file_path
        if full.exists():
            return full.read_text(encoding="utf-8")
        return None

    def list_articles(self, category: str | None = None) -> list[Path]:
        """List all markdown article files in the repo."""
        if category:
            cat_dir = self._path / category
            if cat_dir.exists():
                return sorted(cat_dir.rglob("*.md"))
        return sorted(self._path.rglob("*.md"))

    # ---- git helpers ----

    def log(self, max_count: int = 20) -> list[dict[str, str]]:
        """Return recent commit log."""
        entries: list[dict[str, str]] = []
        for commit in self.repo.iter_commits(max_count=max_count):
            entries.append({
                "hash": commit.hexsha[:12],
                "author": str(commit.author),
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
            })
        return entries

    def diff(self, revision: str = "HEAD~1..HEAD") -> str:
        """Show diff for the given revision range."""
        try:
            return self.repo.git.diff(revision)
        except GitCommandError:
            return "(no diff available)"

    def push(self, remote: str = "origin", branch: str = "main") -> None:
        """Push to remote if configured."""
        try:
            self.repo.remotes[remote].push(branch)
        except (GitCommandError, IndexError) as exc:
            logger.warning("Push failed (no remote configured?): %s", exc)

    # ---- internals ----

    def _commit(self, message: str) -> None:
        try:
            from git import Actor
            author = Actor(self._author_name, self._author_email)
            self.repo.index.commit(
                message,
                author=author,
                committer=author,
            )
        except GitCommandError as exc:
            logger.warning("Commit failed (nothing to commit?): %s", exc)

    def __del__(self) -> None:
        # git.Repo can hold file handles
        if hasattr(self, "_repo") and self._repo is not None:
            try:
                self._repo.close()
            except Exception:
                pass
