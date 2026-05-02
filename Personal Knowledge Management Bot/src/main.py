"""Application entry point — wires everything together and starts the bot.

Usage:
    python -m src.main

Requires TELEGRAM_BOT_TOKEN and at least one LLM provider (OpenAI or Anthropic)
to be configured in the environment or ``.env`` file.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from telegram.ext import Application, ApplicationBuilder

from src.bot.handlers import MessageHandlers
from src.config import settings
from src.storage.session import SessionStore
from src.storage.git_store import GitStore
from src.agent.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy library loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)


def validate_config() -> bool:
    errors: list[str] = []
    if not settings.telegram_bot_token:
        errors.append("TELEGRAM_BOT_TOKEN is required")
    if not settings.has_any_llm:
        errors.append(
            "At least one LLM API key is required:\n"
            "  set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        )

    if errors:
        for e in errors:
            logger.error("Configuration error: %s", e)
        return False

    logger.info(
        "Using LLM provider: %s (mode=%s)",
        settings.effective_provider,
        settings.llm_provider,
    )
    return True


async def main() -> None:
    setup_logging()
    logger.info("Starting Personal Knowledge Management Bot v%s", "0.1.0")

    if not validate_config():
        sys.exit(1)

    # ---- Initialise storage ----
    session_store = SessionStore(settings.database_path)
    session_store.connect()
    logger.info("Session store connected")

    git_store = GitStore(settings.knowledge_base_dir)
    git_store.initialize()
    logger.info("Git knowledge base initialised at %s", settings.knowledge_base_dir)

    # ---- Initialise orchestrator ----
    orchestrator = Orchestrator(session_store)

    # ---- Build Telegram application ----
    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .read_timeout(30)
        .write_timeout(30)
        .get_updates_read_timeout(60)
        .build()
    )

    # ---- Register handlers ----
    handlers = MessageHandlers(
        session_store=session_store,
        git_store=git_store,
        orchestrator=orchestrator,
    )
    handlers.register(app)
    logger.info("Message handlers registered")

    # ---- Pre-load voice model (downloads faster-whisper on first run) ----
    logger.info("Loading voice transcription model…")
    await handlers.voice.startup()

    # ---- Graceful shutdown ----
    shutdown_event = asyncio.Event()

    def _shutdown_handler(*args: object) -> None:
        logger.info("Shutdown signal received — stopping…")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # ---- Start ----
    try:
        async with app:
            logger.info("Bot is running — waiting for messages…")
            await app.start()
            await app.updater.start_polling()

            # Wait for shutdown signal
            await shutdown_event.wait()

            logger.info("Shutting down…")
            await app.updater.stop()
            await app.stop()
    except Exception as exc:
        logger.exception("Fatal error during bot execution: %s", exc)
        sys.exit(1)
    finally:
        session_store.close()
        logger.info("Session store closed")
        logger.info("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
