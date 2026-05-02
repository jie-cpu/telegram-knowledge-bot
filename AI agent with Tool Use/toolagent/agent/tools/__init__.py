"""Tool definitions and schema design for the agent system."""

from agent.tools.base import BaseTool, ToolResult
from agent.tools.registry import ToolRegistry
from agent.tools.calculator import CalculatorTool
from agent.tools.web_search import WebSearchTool
from agent.tools.database_query import DatabaseQueryTool
from agent.tools.code_executor import CodeExecutorTool
from agent.tools.weather import WeatherTool
from agent.tools.file_manager import FileManagerTool
from agent.tools.send_email import SendEmailTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "WebSearchTool",
    "DatabaseQueryTool",
    "CodeExecutorTool",
    "WeatherTool",
    "FileManagerTool",
    "SendEmailTool",
]
