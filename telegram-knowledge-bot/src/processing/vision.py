"""Image analysis via vision-capable LLMs (OpenAI / DeepSeek / Anthropic).

DeepSeek uses an OpenAI-compatible API so requests go through the same
code path with a different base URL.  Local fallback is included when no
vision-capable provider is available.
"""

from __future__ import annotations

import base64
import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

VISION_PROMPT = """You are an image analyst for a personal knowledge management system.
Describe what you see in this image in detail. Focus on:

1. The main subject(s) and what they are doing
2. Any text visible in the image (transcribe it accurately)
3. The context or environment
4. Any diagrams, charts, or data visualizations — describe the data/relationships

Structure your response as a concise but thorough description that could be
used later to write an article about this content."""


class VisionProcessor:
    """Analyse images using a vision-capable LLM.

    Supports:
    - DeepSeek (OpenAI-compatible API, includes vision)
    - OpenAI GPT-4o
    - Anthropic Claude
    """

    def __init__(self) -> None:
        self._provider = settings.effective_provider

    async def describe(self, image_url: str, caption: str | None = None) -> str | None:
        """Download an image and return a text description.

        Args:
            image_url: Pre-signed Telegram file URL.
            caption: Optional user-provided caption.

        Returns:
            Natural-language description, or ``None`` on failure.
        """
        image_bytes = await self._download(image_url)
        if image_bytes is None:
            return None
        return await self._analyse(image_bytes, caption)

    async def _download(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to download image: %s", exc)
            return None

    async def _analyse(self, image_data: bytes, caption: str | None) -> str | None:
        if self._provider == "deepseek":
            return await self._analyse_with_openai_compat(
                image_data, caption,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
            )
        if self._provider == "openai":
            return await self._analyse_with_openai_compat(
                image_data, caption,
                api_key=settings.openai_api_key,
                base_url="https://api.openai.com/v1",
                model=settings.openai_model,
            )
        if self._provider == "anthropic" and settings.has_anthropic:
            return await self._analyse_with_claude(image_data, caption)

        logger.warning("No vision-capable LLM configured — returning placeholder")
        return "[Image analysis unavailable: no vision-capable LLM configured]"

    async def _analyse_with_openai_compat(
        self,
        image_data: bytes,
        caption: str | None,
        api_key: str,
        base_url: str,
        model: str,
    ) -> str | None:
        """Call an OpenAI-compatible vision API (DeepSeek, OpenAI, etc.)."""
        b64 = base64.b64encode(image_data).decode("utf-8")
        content: list[dict] = [
            {"type": "text", "text": VISION_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        if caption:
            content.append({
                "type": "text",
                "text": f"The user also included this caption: {caption}",
            })

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 1024,
                    },
                )
                if resp.status_code == 400:
                    body = resp.text
                    logger.error(
                        "%s vision API rejected the request (model=%s): %s",
                        self._provider, model, body,
                    )
                    return (
                        f"[Image analysis unavailable: {self._provider} model "
                        f"'{model}' does not support vision. "
                        f"Try a different provider or model.]"
                    )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                logger.info(
                    "Image described via %s vision: %.100s…",
                    self._provider, text or "",
                )
                return text
        except httpx.HTTPStatusError as exc:
            logger.error("%s vision API error: %s", self._provider, exc)
            return None

    async def _analyse_with_claude(self, image_data: bytes, caption: str | None) -> str | None:
        b64 = base64.b64encode(image_data).decode("utf-8")
        content: list[dict] = [
            {"type": "text", "text": VISION_PROMPT},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            },
        ]
        if caption:
            content.append({"type": "text", "text": f"User caption: {caption}"})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": settings.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": settings.anthropic_model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": content}],
                    },
                )
                resp.raise_for_status()
                text = resp.json()["content"][0]["text"]
                logger.info("Image described via Claude vision: %.100s…", text)
                return text
        except httpx.HTTPStatusError as exc:
            logger.error("Claude vision API error: %s", exc)
            return None
