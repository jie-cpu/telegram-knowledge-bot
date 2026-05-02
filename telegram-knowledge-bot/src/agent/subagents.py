"""Subagents for parallel processing tasks.

Subagents run concurrently with the main orchestrator to handle expensive or
independent subtasks: fact-checking, external research, entity enrichment,
and cross-referencing against the existing knowledge base.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.models.schemas import ProcessedContent

logger = logging.getLogger(__name__)


class SubagentResult:
    """Result from a subagent execution."""

    def __init__(self, agent_name: str, data: dict[str, Any], error: str | None = None) -> None:
        self.agent_name = agent_name
        self.data = data
        self.error = error


class BaseSubagent:
    """Base class for all subagents."""

    name: str = "base"

    async def run(self, content: ProcessedContent) -> SubagentResult:
        raise NotImplementedError


class EntityEnrichmentAgent(BaseSubagent):
    """Enrich content with additional context — definitions, related concepts."""

    name = "entity-enricher"

    async def run(self, content: ProcessedContent) -> SubagentResult:
        text = content.combined_text
        if not text:
            return SubagentResult(self.name, {"enriched_entities": []})

        entities = content.entities or []
        enriched = []
        for entity in entities[:10]:
            enriched.append({
                "entity": entity,
                "context": f"Mentioned in content about: {text[:100].strip()}...",
            })

        return SubagentResult(self.name, {"enriched_entities": enriched})


class ResearchAgent(BaseSubagent):
    """Perform lightweight external context gathering (simulated).

    In production this would call web search / Wikipedia API to enrich
    content with background information.
    """

    name = "researcher"

    async def run(self, content: ProcessedContent) -> SubagentResult:
        text = content.combined_text
        if not text:
            return SubagentResult(self.name, {"research_notes": []})

        # Extract topic-like phrases (capitalized multi-word sequences)
        # as candidates for external lookups
        import re
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        # Deduplicate
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for c in candidates:
            lower = c.lower()
            if lower not in seen:
                seen.add(lower)
                unique_candidates.append(c)

        research_notes = [
            f"Background context available for: {c}"
            for c in unique_candidates[:5]
        ]

        return SubagentResult(self.name, {"research_notes": research_notes})


class FormattingAgent(BaseSubagent):
    """Prepare content for article generation — structure detection, sectioning."""

    name = "formatter"

    async def run(self, content: ProcessedContent) -> SubagentResult:
        text = content.combined_text
        if not text:
            return SubagentResult(self.name, {"sections": []})

        lines = text.strip().split("\n")
        sections = []
        current_section = "Introduction"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100 and stripped.isupper():
                # Looks like a section heading
                if current_lines:
                    sections.append({
                        "heading": current_section,
                        "content": "\n".join(current_lines).strip(),
                    })
                current_section = stripped
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "heading": current_section,
                "content": "\n".join(current_lines).strip(),
            })

        return SubagentResult(self.name, {"sections": sections})


class SubagentPool:
    """Runs multiple subagents in parallel for a given piece of content."""

    def __init__(self) -> None:
        self._agents: list[BaseSubagent] = [
            EntityEnrichmentAgent(),
            ResearchAgent(),
            FormattingAgent(),
        ]

    async def run_all(self, content: ProcessedContent) -> list[SubagentResult]:
        """Run all subagents concurrently.

        Results are gathered even if some agents fail — failures are captured
        in the individual ``SubagentResult.error`` fields.
        """
        tasks = [agent.run(content) for agent in self._agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final: list[SubagentResult] = []
        for agent, result in zip(self._agents, results):
            if isinstance(result, SubagentResult):
                final.append(result)
            elif isinstance(result, Exception):
                logger.warning("Subagent %s failed: %s", agent.name, result)
                final.append(SubagentResult(agent.name, {}, error=str(result)))
            else:
                final.append(SubagentResult(agent.name, {}, error="Unknown error"))
        return final
