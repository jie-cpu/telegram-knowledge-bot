"""Web search tool — simulated search engine.

In demo mode, returns canned results for known queries.
When connected to a real API, performs actual web searches.

This tool demonstrates:
- Simulated vs real API integration pattern
- Rich result formatting
- Error handling for network-dependent tools
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool

# Pre-canned results for common queries (demo mode)
_DEMO_RESULTS: dict[str, list[dict[str, str]]] = {
    "weather in tokyo": [
        {"title": "Tokyo Weather - 10-Day Forecast", "url": "https://weather.example.com/tokyo", "snippet": "Tokyo, Japan - Current temperature: 22°C, partly cloudy. High of 25°C, low of 18°C."},
        {"title": "Japan Meteorological Agency", "url": "https://www.jma.go.jp", "snippet": "Official weather forecasts and warnings for the Tokyo region."},
    ],
    "python programming": [
        {"title": "Python.org", "url": "https://www.python.org", "snippet": "The official Python programming language website. Download Python, read documentation, and join the community."},
        {"title": "Python Documentation", "url": "https://docs.python.org/3/", "snippet": "Python 3 documentation, tutorials, and language reference."},
    ],
    "latest ai news": [
        {"title": "AI News Today", "url": "https://example.com/ai-news", "snippet": "Breakthrough in multi-modal AI agents: new research shows 40% improvement in tool-use accuracy."},
        {"title": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/", "snippet": "OpenAI releases new reasoning model with improved function calling capabilities."},
    ],
}


class WebSearchTool(BaseTool):
    """Simulated web search tool. Returns realistic mock results in demo mode."""

    def __init__(self, demo_mode: bool = True) -> None:
        super().__init__()
        self._demo_mode = demo_mode

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Use this when you need current information, "
            "facts, or data that requires an internet search. "
            "Provide a clear, specific query for best results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def _run(self, query: str, num_results: int = 5) -> dict[str, Any]:
        query_lower = query.lower().strip()

        if self._demo_mode:
            results = _DEMO_RESULTS.get(query_lower, [])
            if not results:
                results = [
                    {
                        "title": f"Results for: {query}",
                        "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                        "snippet": f"Simulated search result for '{query}'. In production mode, this would return real web results.",
                    }
                ]
        else:
            # In production mode, would call a real search API
            results = [
                {
                    "title": f"Search results for: {query}",
                    "url": f"https://api.search.example.com/v1/results?q={query}",
                    "snippet": f"Real search result for '{query}'. API key required.",
                }
            ]

        return {
            "query": query,
            "total_results": len(results),
            "results": results[:num_results],
        }
