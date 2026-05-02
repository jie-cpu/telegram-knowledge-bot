"""LLM client — unified interface for OpenAI, Anthropic, and mock providers.

Supports:
- OpenAI function calling
- Anthropic tool use
- Mock provider for testing and evaluation
- Token counting and cost tracking
- No LangChain/LangGraph dependency
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "deepseek-chat": {"input": 0.00027, "output": 0.0011},
    "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-2.0-pro": {"input": 0.002, "output": 0.005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "mock-model": {"input": 0.0, "output": 0.0},
}


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMProvider(ABC):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        ...

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        model_key = self.model
        if model_key in _MODEL_COSTS:
            costs = _MODEL_COSTS[model_key]
        else:
            for key in _MODEL_COSTS:
                if model_key.startswith(key):
                    costs = _MODEL_COSTS[key]
                    break
            else:
                return 0.0
        return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 4096):
        super().__init__(model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.llm_api_key or None)
        return self._client

    async def chat(self, messages, tools=None):
        from openai import AsyncOpenAI
        client = self._get_client()
        start = time.perf_counter()
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
        it = response.usage.prompt_tokens if response.usage else 0
        ot = response.usage.completion_tokens if response.usage else 0
        return LLMResponse(content=msg.content, tool_calls=tool_calls, model=self.model, input_tokens=it, output_tokens=ot, cost_usd=self._calculate_cost(it, ot), latency_ms=round(elapsed_ms, 2), finish_reason=choice.finish_reason or "")


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0, max_tokens: int = 4096):
        super().__init__(model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        return self._client

    async def chat(self, messages, tools=None):
        from anthropic import AsyncAnthropic
        client = self._get_client()
        start = time.perf_counter()
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
        response = await client.messages.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        content = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "type": "tool_use", "name": block.name, "input": block.input})
        it = response.usage.input_tokens if response.usage else 0
        ot = response.usage.output_tokens if response.usage else 0
        return LLMResponse(content=content, tool_calls=tool_calls, model=self.model, input_tokens=it, output_tokens=ot, cost_usd=self._calculate_cost(it, ot), latency_ms=round(elapsed_ms, 2), finish_reason=response.stop_reason or "")


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider (OpenAI-compatible)."""

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.0, max_tokens: int = 4096):
        super().__init__(model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            import os
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key or None,
                base_url=settings.deepseek_base_url,
            )
        return self._client

    async def chat(self, messages, tools=None):
        client = self._get_client()
        start = time.perf_counter()
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = await client.chat.completions.create(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            choice = response.choices[0]
            msg = choice.message
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
            it = response.usage.prompt_tokens if response.usage else 0
            ot = response.usage.completion_tokens if response.usage else 0
            return LLMResponse(content=msg.content, tool_calls=tool_calls, model=self.model, input_tokens=it, output_tokens=ot, cost_usd=self._calculate_cost(it, ot), latency_ms=round(elapsed_ms, 2), finish_reason=choice.finish_reason or "")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"DeepSeek API error: {e}")
            return LLMResponse(content=f"Error calling DeepSeek: {e}", tool_calls=[], model=self.model, cost_usd=0, latency_ms=round(elapsed_ms, 2))


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.0, max_tokens: int = 4096):
        super().__init__(model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._client = genai
        return self._client

    def _convert_to_gemini_messages(self, messages):
        """Convert OpenAI-format messages to Gemini format."""
        gemini_history = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    gemini_history.append({"role": "user", "parts": [content]})
            elif m["role"] == "assistant":
                content = m.get("content", "")
                tc = m.get("tool_calls", [])
                parts = []
                if content:
                    parts.append(content)
                if tc:
                    for t in tc:
                        parts.append(f"[Tool call: {t['function']['name']}({t['function']['arguments']})]")
                gemini_history.append({"role": "model", "parts": parts if parts else ["continue"]})
            elif m["role"] == "tool":
                gemini_history.append({"role": "user", "parts": [f"Tool result: {m.get('content', '')}"]})
        return gemini_history, system_instruction

    async def chat(self, messages, tools=None):
        import google.generativeai as genai
        start = time.perf_counter()
        try:
            genai_client = self._get_client()
            history, system_instruction = self._convert_to_gemini_messages(messages)

            model_kwargs = {"model_name": self.model}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            model = genai.GenerativeModel(**model_kwargs)

            # Convert tools to Gemini function declarations
            if tools:
                declarations = []
                for t in tools:
                    func = t.get("function", t)
                    declarations.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", func.get("input_schema", {})),
                    })
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_instruction,
                    tools=[{"function_declarations": declarations}],
                )

            # Rebuild conversation from history
            if len(history) >= 1:
                last = history.pop()
                chat = model.start_chat(history=history)
                response = await chat.send_message_async(last["parts"][0] if isinstance(last["parts"], list) else last["parts"])
            else:
                response = model.generate_content("Hello")

            elapsed_ms = (time.perf_counter() - start) * 1000
            content = response.text if hasattr(response, "text") else str(response)
            tool_calls = []

            # Check for function calls in response
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        fc = part.function_call
                        import json
                        args = {}
                        for k, v in fc.args.items():
                            if hasattr(v, "__getitem__"):
                                try:
                                    args[k] = json.loads(json.dumps(v, default=str))
                                except:
                                    args[k] = str(v)
                            else:
                                args[k] = str(v)
                        tool_calls.append({
                            "id": f"gemini_{fc.name}",
                            "type": "function",
                            "function": {"name": fc.name, "arguments": json.dumps(args)},
                        })

            cost = self._calculate_cost(0, len(content or "") // 4)
            return LLMResponse(content=content, tool_calls=tool_calls, model=self.model, cost_usd=cost, latency_ms=round(elapsed_ms, 2), finish_reason="stop" if not tool_calls else "tool_use")
        except ImportError:
            return LLMResponse(content="Google Generative AI package not installed. Run: pip install google-generativeai", tool_calls=[], model=self.model, cost_usd=0, latency_ms=round((time.perf_counter() - start) * 1000, 2))
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(content=f"Error calling Gemini: {e}", tool_calls=[], model=self.model, cost_usd=0, latency_ms=round(elapsed_ms, 2))


class MockProvider(BaseLLMProvider):
    """Mock LLM — CI/tests only. Keyword matching cannot simulate real AI reasoning.

    A real Agent uses an LLM to UNDERSTAND intent and DECIDE which tool to call.
    This mock is a crude stand-in for running tests without API keys.

    Usage with real LLM:
        LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... toolagent --prompt "..."
        LLM_PROVIDER=gemini GEMINI_API_KEY=... toolagent --prompt "..."
    """

    def __init__(self, model: str = "mock-model"):
        super().__init__(model, temperature=0.0, max_tokens=4096)
        self._tools_used: set[str] = set()

    async def chat(self, messages, tools=None):
        start = time.perf_counter()
        last_role = messages[-1]["role"] if messages else ""

        if last_role == "user":
            self._tools_used.clear()

        # Tool result -> format it nicely
        if last_role == "tool":
            rc = str(messages[-1].get("content", ""))
            prev_tool = ""
            p = messages[-2] if len(messages) >= 2 else None
            if p and p.get("tool_calls"):
                prev_tool = p["tool_calls"][0].get("function", {}).get("name", "")
            try:
                data = json.loads(rc); output = data.get("output", {})
            except: output = {}
            if prev_tool == "calculator":
                ans = f"Answer: {output.get('expression','')} = {output.get('result','')}"
            elif prev_tool == "weather":
                ans = f"Weather in {output.get('city','')}: {output.get('condition','')}, {output.get('temperature','')}, humidity {output.get('humidity','')}, wind {output.get('wind','')}."
            elif prev_tool == "database_query":
                rc2 = output.get("row_count", 0); cols = output.get("columns", [])
                rows = output.get("rows", [])
                ans = f"Query returned {rc2} rows. Columns: {', '.join(cols) if cols else 'none'}."
                if rows: ans += f" First row: {rows[0]}"
            elif prev_tool == "code_executor":
                if output.get("success"): ans = f"Code OK. Output: {str(output.get('stdout',''))[:200]}"
                else: ans = f"Code failed: {output.get('error', output.get('stderr',''))}"
            elif prev_tool == "send_email":
                ans = f"Email sent (mode: {output.get('mode','dry_run')})."
            else: ans = f"Tool '{prev_tool}' completed."
            e = (time.perf_counter() - start) * 1000
            return LLMResponse(content=ans, tool_calls=[], model=self.model,
                input_tokens=0, output_tokens=len(ans), cost_usd=0.0,
                latency_ms=round(e, 2), finish_reason="stop")

        # Minimal keyword fallback (enough for tests, not real AI)
        user_msgs = [m for m in messages if m["role"] == "user" and isinstance(m.get("content"), str)]
        msg = user_msgs[-1].get("content", "").lower() if user_msgs else ""

        selected = None
        if any(w in msg for w in ["weather", "temperature"]):
            import re as _re
            c = _re.search(r"in\s+([\w\s]+?)(?:\?|$|\.)", msg)
            cn = c.group(1).strip() if c else "Unknown"
            selected = ("weather", {"city": cn.title()})
        elif any(w in msg for w in ["product", "customer", "order", "database", "sql", "select"]):
            selected = ("database_query", {"query": "SELECT * FROM products LIMIT 5"})
        elif any(w in msg for w in ["search for", "search the web", "look up", "find about"]):
            selected = ("web_search", {"query": msg, "num_results": 3})
        elif any(w in msg for w in ["email", "mail"]) and "@" in msg:
            selected = ("send_email", {"to": ["user@example.com"], "subject": "Test", "body": "Hello"})
        elif any(w in msg for w in ["code", "python", "execute", "script"]):
            selected = ("code_executor", {"code": "print('Hello from ToolAgent!')"})
        elif any(w in msg for w in ["calculate", "what is", "math", "plus", "minus", "times", "divide", "multiply", "add", "subtract"]):
            nums = re.findall(r"\d+", msg)
            expr = "+".join(nums) if len(nums) >= 2 else ("+".join(nums) if nums else "2+2")
            selected = ("calculator", {"expression": expr})
        elif any(w in msg for w in ["file", "save", "read", "write", "list files", "directory"]):
            selected = ("file_manager", {"operation": "list", "path": "/"})

        tool_calls = []
        if selected and selected[0] not in self._tools_used:
            tn, ta = selected
            if not tools or any(t.get("function", {}).get("name") == tn for t in tools):
                tool_calls.append({"id": f"tc_{tn}", "type": "function",
                    "function": {"name": tn, "arguments": json.dumps(ta)}})
                self._tools_used.add(tn)

        content = None
        if not tool_calls:
            content = (
                "[Mock LLM] No keyword matched.\n\n"
                "For real AI-driven tool selection, connect a real LLM:\n"
                "  echo 'LLM_PROVIDER=deepseek' >> .env\n"
                "  echo 'DEEPSEEK_API_KEY=sk-...' >> .env\n"
                "  PYTHONPATH=. python3 cli/main.py\n"
            )
        e = (time.perf_counter() - start) * 1000
        return LLMResponse(content=content, tool_calls=tool_calls, model=self.model,
            input_tokens=0, output_tokens=len(str(content or tool_calls)), cost_usd=0.0,
            latency_ms=round(e, 2), finish_reason="stop" if content else "tool_use")


class LLMClient:
    """Unified client that supports multiple LLM providers."""

    def __init__(self) -> None:
        provider_name = settings.llm_provider
        self._provider = self._create_provider(provider_name)

    def _create_provider(self, provider_name: str) -> BaseLLMProvider:
        if provider_name == "openai":
            return OpenAIProvider(model=settings.llm_model, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
        elif provider_name == "anthropic":
            return AnthropicProvider(model=settings.anthropic_model, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
        elif provider_name == "deepseek":
            return DeepSeekProvider(model=settings.deepseek_model, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
        elif provider_name == "gemini":
            return GeminiProvider(model=settings.gemini_model, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
        elif provider_name == "mock":
            return MockProvider()
        raise ValueError(f"Unknown provider: {provider_name}")

    async def chat(self, messages, tools=None):
        return await self._provider.chat(messages, tools)

    @property
    def provider_name(self) -> str:
        p = self._provider
        if isinstance(p, OpenAIProvider):
            return f"openai/{p.model}"
        if isinstance(p, AnthropicProvider):
            return f"anthropic/{p.model}"
        if isinstance(p, DeepSeekProvider):
            return f"deepseek/{p.model}"
        if isinstance(p, GeminiProvider):
            return f"gemini/{p.model}"
        return "mock"
