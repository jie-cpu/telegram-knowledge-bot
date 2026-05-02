"""Send email tool — simulates sending email messages.

Demonstrates:
- Confirmation/callback pattern
- Logging-based side effects
- Validation of parameters
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SendEmailTool(BaseTool):
    """Send an email message. In demo mode, logs the message instead of sending."""

    def __init__(self, dry_run: bool = True) -> None:
        super().__init__()
        self._dry_run = dry_run
        self._sent: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return (
            "Send an email to one or more recipients. "
            "Use this when you need to notify users or send information via email. "
            "Supports subject, body, and multiple recipients."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recipient email addresses",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content (plain text)",
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of CC recipients",
                    "default": [],
                },
            },
            "required": ["to", "subject", "body"],
        }

    async def _run(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
    ) -> dict[str, Any]:
        # ── Validation ──────────────────────────────────────────────
        for recipient in to + (cc or []):
            if "@" not in recipient:
                return {
                    "success": False,
                    "error": f"Invalid email address: {recipient}",
                }

        message = {
            "to": to,
            "cc": cc or [],
            "subject": subject,
            "body_preview": body[:200] + ("..." if len(body) > 200 else ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self._dry_run:
            logger.info(f"[DRY RUN] Email sent: to={to}, subject='{subject}'")
            self._sent.append(message)
            return {
                "success": True,
                "mode": "dry_run",
                "message": message,
                "note": "This is a simulation. Enable a real email provider to send actual emails.",
            }
        else:
            # In production, would call an email API
            self._sent.append(message)
            return {
                "success": True,
                "mode": "live",
                "message_id": f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "message": message,
            }
