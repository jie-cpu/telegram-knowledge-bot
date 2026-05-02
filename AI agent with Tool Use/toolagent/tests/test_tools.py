"""Unit tests for all tools."""

import pytest

from agent.tools.base import BaseTool, ToolResult
from agent.tools.calculator import CalculatorTool
from agent.tools.web_search import WebSearchTool
from agent.tools.database_query import DatabaseQueryTool
from agent.tools.code_executor import CodeExecutorTool
from agent.tools.weather import WeatherTool
from agent.tools.file_manager import FileManagerTool
from agent.tools.send_email import SendEmailTool
from agent.tools.registry import ToolRegistry, create_default_registry


class TestCalculatorTool:
    """Tests for the Calculator tool."""

    @pytest.fixture
    def tool(self) -> CalculatorTool:
        return CalculatorTool()

    @pytest.mark.parametrize("expr,expected", [
        ("2 + 2", 4),
        ("10 - 3", 7),
        ("6 * 7", 42),
        ("100 / 4", 25),
        ("2 ** 10", 1024),
        ("(1 + 2) * 3", 9),
    ])
    async def test_basic_arithmetic(self, tool: CalculatorTool, expr: str, expected: float) -> None:
        result = await tool.execute(expression=expr)
        assert result.success
        assert result.output["result"] == expected

    async def test_complex_expression(self, tool: CalculatorTool) -> None:
        result = await tool.execute(expression="(15 * 3) / 5 + 10")
        assert result.success
        assert result.output["result"] == 19.0

    async def test_tool_metadata(self, tool: CalculatorTool) -> None:
        assert tool.name == "calculator"
        assert "mathematical" in tool.description.lower()
        assert "expression" in tool.parameters["required"]
        assert tool.parameters["properties"]["expression"]["type"] == "string"


class TestWebSearchTool:
    """Tests for the Web Search tool."""

    @pytest.fixture
    def tool(self) -> WebSearchTool:
        return WebSearchTool(demo_mode=True)

    async def test_demo_search(self, tool: WebSearchTool) -> None:
        result = await tool.execute(query="weather in tokyo")
        assert result.success
        assert result.output["total_results"] > 0
        assert any("Tokyo" in str(r) for r in result.output["results"])

    async def test_unknown_query(self, tool: WebSearchTool) -> None:
        result = await tool.execute(query="zzzunknownqueryxxx")
        assert result.success
        assert result.output["total_results"] >= 1


class TestDatabaseQueryTool:
    """Tests for the Database Query tool."""

    @pytest.fixture
    def tool(self) -> DatabaseQueryTool:
        return DatabaseQueryTool()

    async def test_show_tables(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="SHOW TABLES")
        assert result.success
        assert "products" in result.output["tables"]
        assert "customers" in result.output["tables"]

    async def test_select_all(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="SELECT * FROM products")
        assert result.success
        assert result.output["row_count"] > 0

    async def test_select_with_where(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="SELECT * FROM products WHERE category = 'Electronics'")
        assert result.success
        assert all(r["category"] == "Electronics" for r in result.output["rows"])

    async def test_describe(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="DESCRIBE products")
        assert result.success
        assert "columns" in result.output

    async def test_count(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="SELECT COUNT(*) FROM products")
        assert result.success

    async def test_sum_aggregate(self, tool: DatabaseQueryTool) -> None:
        result = await tool.execute(query="SELECT SUM(price) FROM products")
        assert result.success


class TestCodeExecutorTool:
    """Tests for the Code Executor tool."""

    @pytest.fixture
    def tool(self) -> CodeExecutorTool:
        return CodeExecutorTool()

    async def test_simple_print(self, tool: CodeExecutorTool) -> None:
        result = await tool.execute(code="print('hello world')")
        assert result.success
        assert "hello world" in result.output["stdout"]

    async def test_math_operation(self, tool: CodeExecutorTool) -> None:
        result = await tool.execute(code="""
import math
print(math.pi)
print(math.sqrt(16))
""")
        assert result.success
        assert "3.14" in result.output["stdout"]
        assert "4.0" in result.output["stdout"]

    async def test_syntax_error(self, tool: CodeExecutorTool) -> None:
        result = await tool.execute(code="print(1/0)")
        assert not result.success

    async def test_dangerous_code_blocked(self, tool: CodeExecutorTool) -> None:
        result = await tool.execute(code="import os; os.system('rm -rf /')")
        assert not result.success

    async def test_list_comprehension(self, tool: CodeExecutorTool) -> None:
        result = await tool.execute(code="print([x**2 for x in range(5)])")
        assert result.success
        assert "[0, 1, 4, 9, 16]" in result.output["stdout"]


class TestWeatherTool:
    """Tests for the Weather tool."""

    @pytest.fixture
    def tool(self) -> WeatherTool:
        return WeatherTool()

    async def test_known_city(self, tool: WeatherTool) -> None:
        result = await tool.execute(city="Tokyo")
        assert result.success
        assert "temperature" in result.output

    async def test_unknown_city(self, tool: WeatherTool) -> None:
        result = await tool.execute(city="UnknownCityXYZ")
        assert result.success  # Still returns suggestions
        assert "error" in result.output or "available_cities" in result.output


class TestFileManagerTool:
    """Tests for the File Manager tool."""

    @pytest.fixture
    def tool(self) -> FileManagerTool:
        return FileManagerTool()

    async def test_write_and_read(self, tool: FileManagerTool) -> None:
        write_result = await tool.execute(operation="write", path="test.txt", content="Hello, World!")
        assert write_result.success

        read_result = await tool.execute(operation="read", path="test.txt")
        assert read_result.success
        assert "Hello, World!" in read_result.output["content"]

    async def test_list_files(self, tool: FileManagerTool) -> None:
        result = await tool.execute(operation="list", path="/")
        assert result.success
        assert "file_count" in result.output

    async def test_path_traversal_blocked(self, tool: FileManagerTool) -> None:
        result = await tool.execute(operation="read", path="../../etc/passwd")
        assert not result.success


class TestSendEmailTool:
    """Tests for the Send Email tool."""

    @pytest.fixture
    def tool(self) -> SendEmailTool:
        return SendEmailTool(dry_run=True)

    async def test_send_email(self, tool: SendEmailTool) -> None:
        result = await tool.execute(
            to=["test@example.com"],
            subject="Test",
            body="Hello from ToolAgent!",
        )
        assert result.success

    async def test_invalid_email(self, tool: SendEmailTool) -> None:
        result = await tool.execute(
            to=["invalid-email"],
            subject="Test",
            body="Body",
        )
        assert not result.success


class TestToolRegistry:
    """Tests for the Tool Registry."""

    def test_create_default(self) -> None:
        registry = create_default_registry()
        assert len(registry) == 7
        assert "calculator" in registry
        assert "web_search" in registry
        assert "database_query" in registry

    def test_tool_formats(self) -> None:
        registry = create_default_registry()
        openai_tools = registry.get_tools_for_openai()
        assert len(openai_tools) == 7
        assert openai_tools[0]["type"] == "function"

        anthropic_tools = registry.get_tools_for_anthropic()
        assert len(anthropic_tools) == 7

    def test_duplicate_registration(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    async def test_execute_unknown(self) -> None:
        registry = create_default_registry()
        result = await registry.execute_tool("nonexistent_tool")
        assert not result.success
        assert "Unknown tool" in (result.error or "")
