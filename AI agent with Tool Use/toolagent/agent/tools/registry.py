"""Tool registry — manages tool discovery, validation, and execution.

The registry acts as a single point of control for all available tools.
It handles:
- Tool registration and discovery
- Tool listing in OpenAI / Anthropic format
- Dispatch to the correct tool based on name
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool with the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    @property
    def all_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def get_tools_for_openai(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function-calling format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def get_tools_for_anthropic(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic tool-use format."""
        return [t.to_anthropic_tool() for t in self._tools.values()]

    async def execute_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with the given parameters."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: '{name}'. Available tools: {', '.join(self._tools.keys())}",
            )
        return await tool.execute(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def create_default_registry(demo_mode: bool = True) -> ToolRegistry:
    """Create a registry with all default tools registered."""
    from agent.tools.calculator import CalculatorTool
    from agent.tools.web_search import WebSearchTool
    from agent.tools.database_query import DatabaseQueryTool
    from agent.tools.code_executor import CodeExecutorTool
    from agent.tools.weather import WeatherTool
    from agent.tools.file_manager import FileManagerTool
    from agent.tools.send_email import SendEmailTool

    registry = ToolRegistry()
    registry.register_all([
        CalculatorTool(),
        WebSearchTool(demo_mode=demo_mode),
        DatabaseQueryTool(),
        CodeExecutorTool(),
        WeatherTool(),
        FileManagerTool(),
        SendEmailTool(dry_run=True),
    ])
    return registry
