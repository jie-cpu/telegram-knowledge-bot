"""Calculator tool — performs arithmetic operations.

This is a simple but essential tool that demonstrates:
- Basic tool definition with JSON Schema
- Deterministic execution for easy testing
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from agent.tools.base import BaseTool

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expression: str) -> float:
    """Evaluate a mathematical expression safely using the AST."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """A calculator tool that safely evaluates mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. "
            "Supports +, -, *, /, **, //, % operators and parentheses. "
            "Examples: '2 + 2', '(15 * 3) / 5', '2 ** 10'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
        }

    async def _run(self, expression: str) -> dict[str, Any]:
        result = _safe_eval(expression)
        return {"expression": expression, "result": result}


class CalculatorToolDemo(CalculatorTool):
    """Alias for backward compatibility."""
    pass
