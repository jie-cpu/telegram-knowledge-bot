"""Agent orchestrator — hand-written async pipeline (no LangGraph).

Processes content through a linear sequence of steps with one branch:
classify → extract → subagents (parallel) → generate/fallback → link.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agent.skills import (
    ClassifySkill,
    ExtractSkill,
    LinkSkill,
    SkillContext,
    SummariseSkill,
)
from src.agent.subagents import SubagentPool
from src.config import settings
from src.models.schemas import Article, MessageType, ProcessedContent
from src.storage.session import SessionStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a personal knowledge management assistant. Your role is to transform raw content into well-structured, informative articles.

Given the user's content, produce a JSON object with these fields:
- "title": A clear, descriptive title (max 80 chars)
- "summary": A 1-2 sentence summary of the content
- "content": The full article body in markdown format. Structure it with sections (## headings), bullet points where appropriate, and clear paragraphs.
- "tags": 3-6 relevant tags as a list of strings

Respond ONLY with the JSON object, no other text."""


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        start = text.find("\n") + 1 if "\n" in text else len(text)
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON response, raw: %.200s", raw)
        return {"title": "Untitled", "content": raw[:2000], "summary": "", "tags": []}


def _build_llm_prompt(
    text: str,
    source_type: str,
    category: str,
    entities: list[str],
    research_notes: list[str],
    sections: list[dict],
) -> str:
    parts = [
        f"Source type: {source_type}",
        f"Suggested category: {category}",
        f"Extracted entities: {', '.join(entities[:15]) or 'none'}",
        "",
        "=== RAW CONTENT ===",
        text[:6000],
        "",
    ]
    if sections:
        parts.append("=== SUGGESTED STRUCTURE ===")
        for s in sections:
            parts.append(f"## {s['heading']}\n{s['content'][:500]}")
    if research_notes:
        parts.append("=== RESEARCH NOTES ===")
        parts.extend(research_notes)
    return "\n".join(parts)


class Orchestrator:
    """Processes content through a linear async pipeline.

    Steps:
        1. classify — categorise content (skill)
        2. extract — extract entities (skill)
        3. subagents — run entity enrichment, research, formatting in parallel
        4. generate — LLM article generation (or fallback if no key configured)
        5. link — cross-reference against existing knowledge-base articles

    No graph framework — just async functions called in sequence.
    """

    def __init__(self, session_store: SessionStore) -> None:
        self._session = session_store
        self._pool = SubagentPool()
        self._skills = {
            "classify": ClassifySkill(),
            "extract": ExtractSkill(),
            "summarise": SummariseSkill(),
            "link-to-existing": LinkSkill(),
        }
        logger.info("Orchestrator ready (provider=%s)", settings.effective_provider)

    async def process(
        self,
        content: ProcessedContent,
        user_id: int,
    ) -> Article | None:
        """Full processing pipeline: classify → extract → subagents → generate → link."""
        start = time.monotonic()
        text = content.combined_text
        if not text:
            logger.warning("Empty content — skipping")
            return None

        existing = self._session.recent_articles(user_id, limit=20)
        ctx = SkillContext(user_id=user_id, existing_articles=existing)

        # 1. Classify
        category = "other"
        classify_result = await self._skills["classify"].execute(ctx, text=text)
        if classify_result.success:
            category = classify_result.output.get("category", "other")

        # 2. Extract entities
        extracted_entities: list[str] = []
        extract_result = await self._skills["extract"].execute(ctx, text=text)
        if extract_result.success:
            extracted_entities = extract_result.output.get("entities", [])

        # 3. Parallel subagents
        sub_results = await self._pool.run_all(content)
        enriched_entities: list[dict] = []
        research_notes: list[str] = []
        sections: list[dict] = []
        for r in sub_results:
            if r.error:
                logger.warning("Subagent %s error: %s", r.agent_name, r.error)
            if r.agent_name == "entity-enricher":
                enriched_entities = r.data.get("enriched_entities", [])
            elif r.agent_name == "researcher":
                research_notes = r.data.get("research_notes", [])
            elif r.agent_name == "formatter":
                sections = r.data.get("sections", [])

        # 4. Generate article (LLM or fallback)
        article: Article | None = None
        if settings.has_any_llm:
            article = await self._generate_with_llm(
                text, content, category, extracted_entities,
                research_notes, sections,
            )
        if article is None:
            article = await self._fallback_generate(text, content, category, extracted_entities)
        if article is None:
            return None

        # 5. Link to existing articles
        link_result = await self._skills["link-to-existing"].execute(ctx, text=article.content)
        if link_result.success:
            links = link_result.output.get("links", [])
            if links:
                article.metadata["related_articles"] = links

        elapsed = time.monotonic() - start
        logger.info(
            "Pipeline processed in %.2fs → article %s", elapsed, article.id[:8],
        )
        return article

    # ── LLM generation ──────────────────────────────────────────────

    async def _generate_with_llm(
        self,
        text: str,
        content: ProcessedContent,
        category: str,
        entities: list[str],
        research_notes: list[str],
        sections: list[dict],
    ) -> Article | None:
        """Route to the configured LLM provider."""
        source_type = (
            content.message_type.value
            if isinstance(content.message_type, MessageType)
            else str(content.message_type)
        )
        prompt = _build_llm_prompt(text, source_type, category, entities, research_notes, sections)
        provider = settings.effective_provider

        if provider == "deepseek":
            return await self._call_openai_compat(
                prompt, content, category,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                provider_name="DeepSeek",
            )
        if provider == "openai":
            return await self._call_openai_compat(
                prompt, content, category,
                api_key=settings.openai_api_key,
                base_url="https://api.openai.com/v1",
                model=settings.openai_model,
                provider_name="OpenAI",
            )
        if provider == "anthropic":
            return await self._call_anthropic(prompt, content, category)

        return None

    async def _call_openai_compat(
        self,
        prompt: str,
        content: ProcessedContent,
        category: str,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str,
    ) -> Article | None:
        """Call any OpenAI-compatible chat API (DeepSeek, OpenAI)."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 2048,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                llm_output = _parse_json_response(raw)
                return Article(
                    title=llm_output.get("title", "Untitled"),
                    content=llm_output.get("content", prompt[:500]),
                    summary=llm_output.get("summary", ""),
                    categories=[category],
                    tags=llm_output.get("tags", []),
                    source_type=content.message_type,
                    source_refs=[content.source_url] if content.source_url else [],
                    metadata={"model": model, "provider": provider_name},
                )
        except Exception as exc:
            logger.error("%s generation failed: %s", provider_name, exc)
            return None

    async def _call_anthropic(
        self,
        prompt: str,
        content: ProcessedContent,
        category: str,
    ) -> Article | None:
        import httpx
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
                        "max_tokens": 2048,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                llm_output = _parse_json_response(resp.json()["content"][0]["text"])
                return Article(
                    title=llm_output.get("title", "Untitled"),
                    content=llm_output.get("content", prompt[:500]),
                    summary=llm_output.get("summary", ""),
                    categories=[category],
                    tags=llm_output.get("tags", []),
                    source_type=content.message_type,
                    source_refs=[content.source_url] if content.source_url else [],
                    metadata={"model": settings.anthropic_model, "provider": "Anthropic"},
                )
        except Exception as exc:
            logger.error("Anthropic generation failed: %s", exc)
            return None

    # ── Fallback (no LLM key) ───────────────────────────────────────

    async def _fallback_generate(
        self,
        text: str,
        content: ProcessedContent,
        category: str,
        entities: list[str],
    ) -> Article | None:
        """Generate article using local skills only."""
        summ = await self._skills["summarise"].execute(SkillContext(user_id=0), text=text)
        summary = summ.output.get("summary", "") if summ.success else ""
        title_words = text.split()[:8]
        title = " ".join(title_words) if title_words else "Untitled"
        if len(title) > 80:
            title = title[:77] + "..."
        return Article(
            title=title,
            content=text[:2000],
            summary=summary,
            categories=[category],
            tags=entities[:5],
            source_type=content.message_type,
            source_refs=[content.source_url] if content.source_url else [],
            metadata={"generated_by": "fallback-skills"},
        )
