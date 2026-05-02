"""Tests for the multi-modal processing modules."""

import pytest

from src.processing.text import TextProcessor
from src.processing.link import LinkProcessor
from src.models.schemas import Article


# ---- Text Processor ----


class TestTextProcessor:
    @pytest.mark.asyncio
    async def test_basic_analysis(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("Hello world, this is a test message.")
        assert result["word_count"] == 7
        assert result["has_question"] is False
        assert result["has_code"] is False
        assert result["has_list"] is False

    @pytest.mark.asyncio
    async def test_detects_code(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("Use `git commit` to save changes.")
        assert result["has_code"] is True

    @pytest.mark.asyncio
    async def test_detects_question(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("What is the capital of France?")
        assert result["has_question"] is True

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("")
        assert result["word_count"] == 0

    @pytest.mark.asyncio
    async def test_list_detection(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("- item one\n- item two\n- item three")
        assert result["has_list"] is True

    @pytest.mark.asyncio
    async def test_cleaned_text(self) -> None:
        proc = TextProcessor()
        result = await proc.analyse("  hello world  ")
        assert result["cleaned_text"] == "hello world"


# ---- Article Model ----


class TestArticle:
    def test_to_markdown_basic(self) -> None:
        article = Article(
            title="Test Article",
            content="This is the body content.",
            summary="A short summary.",
            categories=["technology"],
            tags=["test", "pkm"],
            source_type="text",
        )
        md = article.to_markdown()
        assert "# Test Article" in md
        assert "This is the body content." in md
        assert "#test" in md
        assert "#pkm" in md
        assert "technology" in md

    def test_to_markdown_with_sources(self) -> None:
        article = Article(
            title="Article with Sources",
            content="Body",
            summary="Summary",
            source_type="link",
            source_refs=["https://example.com"],
        )
        md = article.to_markdown()
        assert "https://example.com" in md

    def test_file_path_format(self) -> None:
        article = Article(
            title="My Cool Article Title!",
            content="Body",
            summary="Summary",
            source_type="text",
        )
        path = article.file_path
        # Should be YYYY/MM/hex_title.md
        assert path.endswith(".md")
        assert len(path.split("/")) == 3

    def test_to_markdown_includes_created_date(self) -> None:
        article = Article(
            title="Dated Article",
            content="Body",
            summary="Summary",
            source_type="text",
        )
        md = article.to_markdown()
        assert "Created:" in md
        assert article.created_at.isoformat() in md
