"""Tests for the agent orchestrator, skills, and subagents."""

import pytest

from src.agent.skills import (
    ClassifySkill,
    ExtractSkill,
    SummariseSkill,
    ArticleSkill,
    LinkSkill,
    SkillContext,
)
from src.agent.subagents import (
    EntityEnrichmentAgent,
    ResearchAgent,
    FormattingAgent,
    SubagentPool,
)
from src.models.schemas import Article, ProcessedContent, MessageType


# ---- Skills ----


class TestClassifySkill:
    @pytest.mark.asyncio
    async def test_classify_technology(self) -> None:
        skill = ClassifySkill()
        result = await skill.execute(
            SkillContext(user_id=1),
            text="New Python library for machine learning model deployment",
        )
        assert result.success
        assert result.output["category"] in ClassifySkill.CATEGORIES

    @pytest.mark.asyncio
    async def test_classify_fallback(self) -> None:
        skill = ClassifySkill()
        result = await skill.execute(
            SkillContext(user_id=1),
            text="Random text with no clear category xyzzy",
        )
        assert result.success
        assert result.output["category"] == "other"


class TestExtractSkill:
    @pytest.mark.asyncio
    async def test_extract_entities(self) -> None:
        skill = ExtractSkill()
        result = await skill.execute(
            SkillContext(user_id=1),
            text="Apple released the new MacBook Pro with M3 chip.",
        )
        assert result.success
        assert len(result.output["entities"]) > 0


class TestSummariseSkill:
    @pytest.mark.asyncio
    async def test_summarise(self) -> None:
        skill = SummariseSkill()
        result = await skill.execute(
            SkillContext(user_id=1),
            text="First sentence about something. Second sentence about something else. "
                 "Third sentence. Fourth sentence.",
        )
        assert result.success
        assert "First sentence" in result.output["summary"]


class TestArticleSkill:
    @pytest.mark.asyncio
    async def test_create_article(self) -> None:
        skill = ArticleSkill()
        result = await skill.execute(
            SkillContext(user_id=1),
            title="Test",
            content_body="Content",
            categories=["tech"],
            tags=["test"],
            source_type="text",
        )
        assert result.success
        article = result.output["article"]
        assert isinstance(article, Article)
        assert article.title == "Test"


class TestLinkSkill:
    @pytest.mark.asyncio
    async def test_link_to_existing(self) -> None:
        skill = LinkSkill()
        ctx = SkillContext(
            user_id=1,
            existing_articles=[
                {"article_id": "1", "title": "Python Tips and Tricks", "file_path": "path1"},
                {"article_id": "2", "title": "Machine Learning Basics", "file_path": "path2"},
            ],
        )
        result = await skill.execute(
            ctx,
            text="Here are some Python tips for machine learning",
        )
        assert result.success
        assert len(result.output["links"]) > 0


# ---- Subagents ----


@pytest.fixture
def sample_content() -> ProcessedContent:
    return ProcessedContent(
        message_id="test123",
        message_type=MessageType.TEXT,
        raw_text="Artificial Intelligence and Machine Learning are transforming "
                 "the way we build software. Large Language Models like GPT-4 "
                 "can understand and generate human-like text.",
        entities=["Artificial Intelligence", "Machine Learning", "GPT-4"],
    )


class TestEntityEnrichmentAgent:
    @pytest.mark.asyncio
    async def test_enriches_entities(self, sample_content: ProcessedContent) -> None:
        agent = EntityEnrichmentAgent()
        result = await agent.run(sample_content)
        assert result.error is None
        assert len(result.data["enriched_entities"]) > 0


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_research_notes(self, sample_content: ProcessedContent) -> None:
        agent = ResearchAgent()
        result = await agent.run(sample_content)
        assert result.error is None
        assert "research_notes" in result.data


class TestFormattingAgent:
    @pytest.mark.asyncio
    async def test_detects_sections(self) -> None:
        content = ProcessedContent(
            message_id="test456",
            message_type=MessageType.TEXT,
            raw_text="INTRODUCTION\nThis is intro content.\n\nMETHODS\n"
                     "This is methods content.\n\nRESULTS\nThis is results content.",
        )
        agent = FormattingAgent()
        result = await agent.run(content)
        assert result.error is None
        assert len(result.data["sections"]) > 0


class TestSubagentPool:
    @pytest.mark.asyncio
    async def test_runs_all_agents(self, sample_content: ProcessedContent) -> None:
        pool = SubagentPool()
        results = await pool.run_all(sample_content)
        assert len(results) == 3  # entity-enricher, researcher, formatter
        for r in results:
            assert r.error is None
