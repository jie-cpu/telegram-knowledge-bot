"""Skills / commands pattern for the content-organisation agent.

Each *skill* is a self-contained capability the orchestrator can invoke:
classify, extract-entities, summarise, write-article, link-to-existing.
Skills accept structured input and return structured output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillContext:
    """Context passed to every skill invocation."""
    user_id: int
    existing_articles: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillResult:
    success: bool
    output: Any = None
    error: str | None = None


class Skill:
    """Base class for all skills."""

    name: str = "base"
    description: str = ""

    async def execute(self, context: SkillContext, **kwargs: Any) -> SkillResult:
        raise NotImplementedError


class ClassifySkill(Skill):
    """Classify content into categories (technology, science, personal, etc.)."""

    name = "classify"
    description = "Classify content into knowledge-domain categories"

    CATEGORIES = [
        "technology", "science", "programming", "ai-ml",
        "health", "personal", "productivity", "business",
        "education", "arts", "news", "other",
    ]

    async def execute(self, context: SkillContext, *, text: str, **kwargs: Any) -> SkillResult:
        # Use keyword matching + heuristics as a fast pre-classifier.
        # The orchestrator LLM provides the real classification; this is a
        # fallback that works even during local development.
        text_lower = text.lower()
        scores: dict[str, float] = {c: 0.0 for c in self.CATEGORIES}

        keywords: dict[str, list[str]] = {
            "technology": ["tech", "software", "hardware", "digital", "computer"],
            "programming": ["python", "code", "api", "function", "algorithm", "bug", "deploy"],
            "ai-ml": ["machine learning", "deep learning", "llm", "neural", "gpt", "model", "training"],
            "science": ["research", "study", "experiment", "hypothesis", "scientific"],
            "health": ["health", "medical", "symptom", "treatment", "exercise", "diet"],
            "productivity": ["productivity", "workflow", "efficiency", "time management", "habit"],
            "business": ["business", "startup", "market", "revenue", "strategy"],
            "education": ["learn", "course", "tutorial", "lesson", "study"],
            "arts": ["art", "music", "design", "creative", "photography"],
            "news": ["report", "breaking", "announce", "latest"],
        }

        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    scores[category] += 1.0

        # pick top category
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            best = "other"

        return SkillResult(success=True, output={"category": best, "scores": scores})


class ExtractSkill(Skill):
    """Extract entities, keywords, and topics from content."""

    name = "extract"
    description = "Extract named entities and key topics"

    async def execute(self, context: SkillContext, *, text: str, **kwargs: Any) -> SkillResult:
        # Keyword-based extraction as fallback; the LLM does the real work.
        import re

        # Simple keyword extraction: find capitalized phrases
        potential_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        # Remove very short or generic
        stopwords = {"This", "That", "The", "What", "How", "Why", "When", "Where",
                     "Also", "However", "Therefore", "Furthermore", "Additionally"}
        entities = [e for e in potential_entities if len(e) > 3 and e not in stopwords]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_entities: list[str] = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)

        return SkillResult(success=True, output={
            "entities": unique_entities[:20],
            "entity_count": len(unique_entities),
        })


class SummariseSkill(Skill):
    """Create a concise summary of the processed content."""

    name = "summarise"
    description = "Summarise content into 2-3 sentences"

    async def execute(self, context: SkillContext, *, text: str, max_sentences: int = 3, **kwargs: Any) -> SkillResult:
        # Truncation-based fallback; the orchestrator LLM provides the real summary.
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        summary_sentences = sentences[:max_sentences]
        summary = ". ".join(summary_sentences) + "."
        return SkillResult(success=True, output={"summary": summary})


class LinkSkill(Skill):
    """Find connections between new content and existing articles in the KB."""

    name = "link-to-existing"
    description = "Find links between new content and existing knowledge-base articles"

    async def execute(self, context: SkillContext, *, text: str, **kwargs: Any) -> SkillResult:
        if not context.existing_articles:
            return SkillResult(success=True, output={"links": []})

        # Simple keyword overlap matching
        new_words = set(w.lower() for w in text.split() if len(w) > 4)
        links = []
        for article in context.existing_articles:
            title_words = set(article.get("title", "").lower().split())
            overlap = new_words & title_words
            if overlap:
                links.append({
                    "article_id": article.get("article_id"),
                    "title": article.get("title"),
                    "relevance": len(overlap),
                    "file_path": article.get("file_path"),
                })

        # Sort by relevance
        links.sort(key=lambda x: x["relevance"], reverse=True)
        return SkillResult(success=True, output={"links": links[:5]})
