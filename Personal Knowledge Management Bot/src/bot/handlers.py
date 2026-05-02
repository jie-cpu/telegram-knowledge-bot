"""Telegram message handlers — the entry point for all user interactions.

Handles voice notes, images/photos, links, text messages, documents, and
commands (/start, /help, /recent, /search, /stats, /export).  Every message is
classified, rate-checked, enqueued, and acknowledged to the user.
"""

from __future__ import annotations

import logging
import re

from telegram import Document, PhotoSize, Update, Voice
from telegram.constants import MessageLimit
from telegram.ext import CommandHandler, ContextTypes, MessageHandler, filters

from src.config import settings
from src.models.schemas import IncomingMessage, MessageType, ProcessedContent
from src.processing.link import LinkProcessor
from src.processing.text import TextProcessor
from src.processing.vision import VisionProcessor
from src.processing.voice import VoiceProcessor
from src.queue.rate_limiter import TokenBucket
from src.storage.git_store import GitStore
from src.storage.session import SessionStore
from src.agent.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


class MessageHandlers:
    """Register all Telegram handlers and owns the processing pipeline."""

    def __init__(
        self,
        session_store: SessionStore,
        git_store: GitStore,
        orchestrator: Orchestrator,
    ) -> None:
        self._session = session_store
        self._git = git_store
        self._orchestrator = orchestrator
        self._rate_limiter = TokenBucket(
            capacity=settings.rate_limit_messages,
            window_seconds=settings.rate_limit_window,
        )
        self._voice_processor = VoiceProcessor()
        self._vision_processor = VisionProcessor()
        self._link_processor = LinkProcessor()
        self._text_processor = TextProcessor()

    @property
    def voice(self) -> VoiceProcessor:
        return self._voice_processor

    # ---- register ----

    def register(self, app: "Application") -> None:  # noqa: F821
        """Add all handlers to the Application."""
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("recent", self.cmd_recent))
        app.add_handler(CommandHandler("search", self.cmd_search))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("export", self.cmd_export))

        app.add_handler(MessageHandler(filters.VOICE, self.on_voice))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.on_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))

    # ---- commands ----

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_user and update.effective_chat
        user = update.effective_user
        self._session.set(user.id, "onboarded", True)
        await update.message.reply_text(
            f"Welcome, {user.first_name}! I'm your Personal Knowledge Management Bot.\n\n"
            "Send me:\n"
            "• Voice notes — I'll transcribe and save them\n"
            "• Images — I'll analyse and describe them\n"
            "• Links — I'll extract and summarise the content\n"
            "• Text — I'll organise it into structured articles\n\n"
            "Commands:\n"
            "/recent — View your recent articles\n"
            "/search <query> — Search your knowledge base\n"
            "/stats — Usage statistics\n"
            "/export — Push knowledge base to git remote\n"
            "/help — Detailed help",
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "PKM Bot Help\n\n"
            "I process multi-modal input and organise it into structured articles "
            "stored in a git-backed knowledge base.\n\n"
            "**Multi-modal input:**\n"
            "• 🎤 Voice notes → Whisper transcription → Article\n"
            "• 🖼 Images → Vision AI description → Article\n"
            "• 🔗 Links → Content extraction → Article\n"
            "• 📝 Text → Direct analysis → Article\n\n"
            "**Commands:**\n"
            "/recent — Last 10 articles\n"
            "/search <query> — Full-text search\n"
            "/stats — Token usage and article count\n"
            "/export — Push KB to remote git\n\n"
            "**Rate limits:** {}/{} messages per minute per user.".format(
                settings.rate_limit_messages,
                settings.rate_limit_window,
            ),
        )

    async def cmd_recent(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_user
        articles = self._session.recent_articles(update.effective_user.id, limit=10)
        if not articles:
            await update.message.reply_text("No articles yet. Send me something to process!")
            return

        lines = ["**Your Recent Articles:**\n"]
        for a in articles:
            date = a["created_at"][:10] if a["created_at"] else "unknown"
            lines.append(f"• [{a['title']}]({a['file_path']}) — {date} ({a['source_type']})")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text("Usage: /search <your query>")
            return

        results = self._session.search_articles(query, limit=5)
        if not results:
            await update.message.reply_text(f"No articles found matching '{query}'")
            return

        lines = [f"**Results for '{query}':**\n"]
        for r in results:
            lines.append(f"• [{r['title']}]({r['file_path']}) — {r['summary'][:100]}…")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_user
        # Count articles from the file system
        articles = self._git.list_articles()
        kb_size = sum(f.stat().st_size for f in articles) if articles else 0

        commit_count = len(self._git.log(max_count=100))

        await update.message.reply_text(
            f"**Knowledge Base Stats**\n\n"
            f"Articles: {len(articles)}\n"
            f"Commits: {commit_count}\n"
            f"Storage: {kb_size / 1024:.1f} KB",
            parse_mode="Markdown",
        )

    async def cmd_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Pushing to remote git…")
        try:
            self._git.push()
            await update.message.reply_text("Push complete!")
        except Exception as exc:
            logger.error("Push failed: %s", exc)
            await update.message.reply_text(f"Push failed: {exc}")

    # ---- message handlers ----

    async def on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.message and update.effective_user
        voice: Voice = update.message.voice
        file = await voice.get_file()
        file_url = file.file_path

        msg = IncomingMessage(
            telegram_message_id=update.message.message_id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            message_type=MessageType.VOICE,
            file_id=voice.file_id,
            file_unique_id=voice.file_unique_id,
            caption=update.message.caption,
        )

        await self._enqueue_and_ack(update, msg, "🎤 Voice note received — transcribing…")

        # Process inline: download & transcribe immediately
        text = await self._voice_processor.transcribe(file_url)
        if text:
            await self._process_content(msg, text, update)
        else:
            await update.message.reply_text("❌ Could not transcribe voice note.")

    async def on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.message and update.effective_user
        photos: list[PhotoSize] = update.message.photo
        best_photo = max(photos, key=lambda p: p.file_size or 0)
        file = await best_photo.get_file()
        file_url = file.file_path

        msg = IncomingMessage(
            telegram_message_id=update.message.message_id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            message_type=MessageType.IMAGE,
            file_id=best_photo.file_id,
            file_unique_id=best_photo.file_unique_id,
            caption=update.message.caption,
        )

        await self._enqueue_and_ack(update, msg, "🖼 Image received — analysing…")

        text = await self._vision_processor.describe(file_url, update.message.caption)
        if text:
            await self._process_content(msg, text, update)
        else:
            await update.message.reply_text("❌ Could not analyse image.")

    async def on_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.message and update.effective_user
        doc: Document = update.message.document
        file = await doc.get_file()
        file_url = file.file_path

        msg = IncomingMessage(
            telegram_message_id=update.message.message_id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            message_type=MessageType.DOCUMENT,
            file_id=doc.file_id,
            file_unique_id=doc.file_unique_id,
            caption=update.message.caption,
        )

        await self._enqueue_and_ack(
            update, msg,
            f"📄 Document '{doc.file_name}' received — processing…",
        )

        # For now treat documents as text if we can read them;
        # in production add PDF/DOCX parsing.
        text = update.message.caption or f"[Document: {doc.file_name}]"
        await self._process_content(msg, text, update)

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.message and update.effective_user
        text = update.message.text.strip()
        if not text:
            return

        # Detect links
        urls = URL_RE.findall(text)
        msg_type = MessageType.LINK if urls else MessageType.TEXT

        msg = IncomingMessage(
            telegram_message_id=update.message.message_id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            message_type=msg_type,
            text=text,
        )

        await self._enqueue_and_ack(
            update, msg,
            "🔗 Link received — extracting content…" if urls else "📝 Text received — processing…",
        )

        if urls and msg_type == MessageType.LINK:
            # Try to extract content from the first URL
            url = urls[0]
            extracted = await self._link_processor.extract(url)
            if extracted:
                enriched_text = f"Source: {url}\n\n{extracted}"
                if len(urls) > 1:
                    enriched_text += f"\n\n[+ {len(urls) - 1} more links]"
                text = enriched_text

        await self._process_content(msg, text, update)

    # ---- internal helpers ----

    async def _enqueue_and_ack(
        self,
        update: Update,
        msg: IncomingMessage,
        ack_text: str,
    ) -> None:
        """Check rate limit, enqueue, and acknowledge to user."""
        if not self._rate_limiter.is_allowed(msg.telegram_user_id):
            remaining = self._rate_limiter.remaining(msg.telegram_user_id)
            await update.message.reply_text(
                f"⏳ Rate limit reached. {remaining} messages remaining in this window."
            )
            return

        await update.message.reply_text(ack_text)
        # Log to session store
        self._session.log_processing(
            msg.id, msg.telegram_user_id, msg.message_type.value,
        )

    async def _process_content(self, msg: IncomingMessage, text: str, update: Update) -> None:
        """Run the full processing pipeline for content that has been transcribed/extracted."""
        # 1. Analyse text metadata
        analysis = await self._text_processor.analyse(text)

        # 2. Build processed content
        processed = ProcessedContent(
            message_id=msg.id,
            message_type=msg.message_type,
            raw_text=text if msg.message_type == MessageType.TEXT else None,
            transcription=text if msg.message_type == MessageType.VOICE else None,
            vision_description=text if msg.message_type == MessageType.IMAGE else None,
            extracted_text=text if msg.message_type == MessageType.LINK else None,
            source_url=URL_RE.findall(text)[0] if URL_RE.findall(text) else None,
        )

        # 3. Agent orchestrator
        article = await self._orchestrator.process(processed, msg.telegram_user_id)
        if article is None:
            await update.message.reply_text("❌ Could not organise content into an article.")
            return

        # 4. Save to git knowledge base
        file_path = self._git.save_article(article)

        # 5. Index in session store
        self._session.index_article(
            article_id=article.id,
            title=article.title,
            summary=article.summary,
            categories=article.categories,
            tags=article.tags,
            file_path=str(file_path),
            source_type=article.source_type.value,
            user_id=msg.telegram_user_id,
        )

        # 6. Log completion
        self._session.log_processing(
            msg.id, msg.telegram_user_id, msg.message_type.value,
        )

        # 7. Reply to user
        preview = article.summary[:MessageLimit.MAX_TEXT_LENGTH - 500]
        tags_str = " ".join(f"#{t}" for t in article.tags[:5]) if article.tags else ""
        await update.message.reply_text(
            f"✅ **Article saved!**\n\n"
            f"**{article.title}**\n\n"
            f"{preview}\n\n"
            f"📁 `{article.file_path}`\n"
            f"{tags_str}",
            parse_mode="Markdown",
        )
