"""Conversation memory — maintains context across agent steps.

Manages:
- Message history with system prompt
- Token-aware context window management
- Summarization for long conversations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMemory:
    """Manages conversation history with context window management."""

    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 128_000
    _tool_results: dict[str, Any] = field(default_factory=dict)

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = prompt

    def add_message(self, role: str, content: str | list | None, **kwargs: Any) -> dict[str, Any]:
        """Add a message to the conversation history."""
        message: dict[str, Any] = {"role": role, "content": content or ""}
        message.update(kwargs)
        self.messages.append(message)
        return message

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: Any) -> dict[str, Any]:
        """Add a tool result message to the conversation history."""
        self._tool_results[tool_call_id] = result

        # Format for OpenAI-style tool responses
        content_str = str(result)
        if isinstance(result, dict):
            import json
            content_str = json.dumps(result, indent=2, default=str)

        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content_str,
        }
        self.messages.append(message)
        return message

    def add_tool_result_anthropic(self, tool_call_id: str, tool_name: str, result: Any) -> dict[str, Any]:
        """Add a tool result formatted for Anthropic's API."""
        content_str = str(result)
        if isinstance(result, dict):
            import json
            content_str = json.dumps(result, indent=2, default=str)

        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content_str,
                }
            ],
        }
        self.messages.append(message)
        return message

    def get_messages_for_openai(self) -> list[dict[str, Any]]:
        """Get messages formatted for OpenAI API."""
        result: list[dict[str, Any]] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(self.messages)
        return result

    def get_messages_for_anthropic(self) -> list[dict[str, Any]]:
        """Get messages formatted for Anthropic API."""
        result: list[dict[str, Any]] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt},)
        result.extend(self.messages)
        return result

    def get_last_message(self) -> dict[str, Any] | None:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None

    def get_tool_result(self, tool_call_id: str) -> Any | None:
        """Get the result of a specific tool call."""
        return self._tool_results.get(tool_call_id)

    def clear(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self.messages.clear()
        self._tool_results.clear()

    def count_messages(self) -> int:
        """Count the number of messages in history."""
        return len(self.messages)

    def summarize(self) -> dict[str, Any]:
        """Get a summary of the conversation for logging/debugging."""
        return {
            "message_count": len(self.messages),
            "tool_call_count": len(self._tool_results),
            "first_message": self.messages[0] if self.messages else None,
            "last_message": self.messages[-1] if self.messages else None,
        }

    @staticmethod
    def create_default_system_prompt(tool_names: list[str]) -> str:
        """Create a default system prompt with tool descriptions."""
        tools_str = ", ".join(sorted(tool_names))
        return f"""You are ToolAgent, an AI assistant with access to the following tools: {tools_str}.

Your capabilities:
- You can use tools to accomplish tasks step by step
- You should choose the right tool for each step
- You can use multiple tools in sequence to solve complex problems
- If a tool returns an error, try an alternative approach or tool
- When you have enough information, provide a clear answer to the user

Guidelines:
- Plan your approach before calling tools
- Use the most specific tool for each task
- When searching, use targeted queries
- For calculations, use the calculator tool for precision
- Never fabricate tool results — trust what the tools return
- If you cannot complete a task with available tools, say so clearly
"""
