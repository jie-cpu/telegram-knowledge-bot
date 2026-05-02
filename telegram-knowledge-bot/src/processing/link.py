"""Link / URL content extraction using trafilatura."""

from __future__ import annotations

import logging

import httpx
import trafilatura

logger = logging.getLogger(__name__)


class LinkProcessor:
    """Fetch a URL and extract its main content using trafilatura."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PKMKnowledgeBot/1.0; "
                    "+https://github.com/your-org/personal-knowledge-bot)"
                ),
            },
        )

    async def extract(self, url: str) -> str | None:
        """Fetch *url* and return the extracted main content as plain text.

        Returns ``None`` if the page cannot be reached or has no extractable
        content.
        """
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch URL %s: %s", url, exc)
            return None

        text = trafilatura.extract(html, include_links=True, include_images=False,
                                   output_format="markdown")
        if text:
            logger.info("Extracted %d chars from %s", len(text), url)
            return text.strip()

        logger.warning("trafilatura returned no content for %s", url)
        return None

    async def close(self) -> None:
        await self._client.aclose()
