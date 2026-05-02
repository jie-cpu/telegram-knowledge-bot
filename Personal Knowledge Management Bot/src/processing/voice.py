"""Voice note transcription — local (faster-whisper) or API (OpenAI Whisper).

By default uses **local** faster-whisper so no API key is needed.
Set ``transcription_mode=api`` in config to use OpenAI Whisper instead.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.config import settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """Transcribe voice notes — local faster-whisper or OpenAI Whisper API."""

    def __init__(self) -> None:
        self._api_key = settings.openai_api_key
        self._mode = settings.transcription_mode
        self._model_size = settings.whisper_model_size
        self._local_model: WhisperModel | None = None
        self._model_ready = False

    async def startup(self) -> None:
        """Pre-load the Whisper model at bot startup (not lazily)."""
        if self._mode == "api" and self._api_key:
            logger.info("Using OpenAI Whisper API — no local model to load")
            self._model_ready = True
            return

        model = await self._get_local_model()
        if model is not None:
            self._model_ready = True
            logger.info("Whisper model '%s' ready", self._model_size)
        else:
            logger.error(
                "Failed to load Whisper model '%s'. "
                "Voice notes will fall back to a text placeholder.",
                self._model_size,
            )

    @property
    def is_ready(self) -> bool:
        return self._model_ready

    async def transcribe(self, file_url: str) -> str | None:
        """Download a voice file and transcribe it.

        Args:
            file_url: Pre-signed Telegram file URL.

        Returns:
            Transcribed text, or ``None`` on failure.
        """
        if self._mode != "api" and not self._model_ready:
            logger.warning("Voice transcription requested but model not ready yet")
            return None

        audio_bytes = await self._download(file_url)
        if audio_bytes is None:
            return None

        if self._mode == "api" and self._api_key:
            return await self._transcribe_api(audio_bytes)
        else:
            return await self._transcribe_local(audio_bytes)

    async def _download(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to download voice file: %s", exc)
            return None

    # ---- Local transcription (faster-whisper) ----

    async def _transcribe_local(self, audio_data: bytes) -> str | None:
        """Transcribe using local faster-whisper model."""
        suffix = ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            model = await self._get_local_model()
            if model is None:
                return "[Voice transcription unavailable: faster-whisper not installed]"

            # Run synchronous transcribe in a thread to avoid blocking
            import asyncio

            def _run_transcribe() -> tuple[list[str], float | None]:
                segments, info = model.transcribe(tmp_path, language=None)
                texts = [seg.text.strip() for seg in segments]
                return texts, info.duration

            text_parts, duration = await asyncio.to_thread(_run_transcribe)
            result = " ".join(text_parts)

            if result:
                logger.info(
                    "Local Whisper transcribed voice (%.1fs audio): %.100s…",
                    duration or 0, result,
                )
                return result
            logger.warning("Local Whisper returned empty transcription")
            return None

        except Exception as exc:
            logger.error("Local transcription failed: %s", exc)
            return None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def _get_local_model(self) -> WhisperModel | None:
        """Lazy-load the faster-whisper model (cached)."""
        if self._local_model is not None:
            return self._local_model

        try:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model '%s'…", self._model_size)
            # Run model loading in a thread to avoid blocking the event loop
            import asyncio
            model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=2,
            )
            self._local_model = model
            logger.info("faster-whisper model '%s' loaded", self._model_size)
            return model
        except ImportError:
            logger.error(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
            return None
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            return None

    # ---- API transcription (OpenAI Whisper) ----

    async def _transcribe_api(self, audio_data: bytes) -> str | None:
        """Transcribe using OpenAI Whisper API."""
        suffix = ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(tmp_path, "rb") as f:
                    files = {"file": (f"voice{suffix}", f, "audio/ogg")}
                    data = {"model": "whisper-1", "language": None}
                    resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        files=files,
                        data=data,
                    )
                resp.raise_for_status()
                result = resp.json()
                text = result.get("text", "").strip()
                if text:
                    logger.info("Whisper API transcribed voice: %.100s…", text)
                    return text
                logger.warning("Whisper API returned empty transcription")
                return None
        except httpx.HTTPStatusError as exc:
            logger.error("Whisper API error: %s", exc)
            return None
        finally:
            Path(tmp_path).unlink(missing_ok=True)
