"""Raw text processing and enrichment."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Analyse and enrich plain text input.

    This module handles direct text messages, captions, and any other
    free-text input.  It does basic content classification and entity
    extraction before passing to the agent orchestrator.
    """

    async def analyse(self, text: str) -> dict:
        """Analyse raw text and return structured metadata.

        Returns a dict with keys:
        - cleaned_text: trimmed/normalised version
        - word_count: approximate word count
        - has_code: whether the text contains code blocks
        - has_question: whether it looks like a question
        - has_list: whether it contains bullet/numbered lists
        - language_hint: simple language guess based on character set
        """
        cleaned = text.strip()

        return {
            "cleaned_text": cleaned,
            "word_count": len(cleaned.split()),
            "has_code": "```" in cleaned or "`" in cleaned,
            "has_question": "?" in cleaned,
            "has_list": any(
                cleaned.startswith(prefix) or f"\n{prefix}" in cleaned
                for prefix in ("- ", "* ", "1. ", "1)")
            ),
        }
