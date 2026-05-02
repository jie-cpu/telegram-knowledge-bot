"""Pydantic data models used across the application."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    VOICE = "voice"
    IMAGE = "image"
    LINK = "link"
    TEXT = "text"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IncomingMessage(BaseModel):
    """Raw message received from Telegram."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    telegram_message_id: int
    telegram_user_id: int
    chat_id: int
    message_type: MessageType
    text: str | None = None
    file_id: str | None = None
    file_unique_id: str | None = None
    caption: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessedContent(BaseModel):
    """Content after initial multi-modal processing."""

    message_id: str
    message_type: MessageType
    raw_text: str | None = None
    transcription: str | None = None  # voice → text
    vision_description: str | None = None  # image → description
    extracted_text: str | None = None  # link → content
    entities: list[str] = Field(default_factory=list)
    summary: str | None = None
    categories: list[str] = Field(default_factory=list)
    source_url: str | None = None  # for links
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def combined_text(self) -> str:
        """Merge all available text sources."""
        parts = [
            self.transcription,
            self.vision_description,
            self.extracted_text,
            self.raw_text,
        ]
        return "\n".join(p for p in parts if p)


class Article(BaseModel):
    """Structured article produced by the agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    content: str  # markdown body
    summary: str
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_type: MessageType
    source_refs: list[str] = Field(default_factory=list)  # links / file ids
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the article as a markdown file suitable for git storage."""
        lines = [
            f"# {self.title}",
            "",
            f"> {self.summary}",
            "",
            "---",
            "",
            self.content,
            "",
            "---",
            "",
        ]
        if self.tags:
            lines.append(f"**Tags:** {', '.join(f'#{t}' for t in self.tags)}")
        if self.categories:
            lines.append(f"**Categories:** {', '.join(self.categories)}")
        lines.append(f"**Created:** {self.created_at.isoformat()}")
        if self.source_refs:
            lines.append(f"**Sources:** {', '.join(self.source_refs)}")
        lines.append("")
        return "\n".join(lines)

    @property
    def file_path(self) -> str:
        """Relative path inside the knowledge base."""
        date_prefix = self.created_at.strftime("%Y/%m")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.title)
        safe_title = safe_title.strip().replace(" ", "_")[:80]
        return f"{date_prefix}/{self.id[:8]}_{safe_title}.md"


class ProcessingResult(BaseModel):
    """Final result sent back to the user."""

    article: Article | None = None
    status: ProcessingStatus
    error: str | None = None
    processing_time_ms: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
