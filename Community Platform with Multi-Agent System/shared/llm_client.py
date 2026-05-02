"""
Multi-provider LLM client with config-driven model selection.

Supports: OpenAI, Anthropic, Ollama (local models).
Reads configuration from config.yaml and environment variables.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def load_env_file(env_path: str | Path) -> None:
    """Load .env file without requiring python-dotenv.

    Reads key=value lines, skips comments and blank lines.
    Supports quoted values and inline comments.
    Merges into os.environ (does NOT overwrite existing vars).
    """
    env_path = Path(env_path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            # Split on first =
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            # Strip quotes from value
            value = value.strip().strip("\"'")
            # Strip inline comments (after a space + #)
            if " #" in value:
                value = value.split(" #")[0].strip()
            # Don't overwrite existing env vars (user's export takes precedence)
            if key not in os.environ:
                os.environ[key] = value


# Auto-load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_env_file(_env_path)


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class APIKeyMissingError(LLMError):
    """API key not configured for the requested provider."""


class ProviderTimeoutError(LLMError):
    """LLM provider timed out."""


class ResponseParseError(LLMError):
    """Failed to parse structured output from LLM."""


class LLMClient:
    """Unified client for OpenAI, Anthropic, DeepSeek, Gemini, and local (Ollama) models.

    Usage:
        client = LLMClient.from_config("config.yaml")
        response = client.chat("Write a haiku about coding")
        tasks = client.structured_chat(prompt, schema={"tasks": [...]})
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Token & cost tracking
        self._last_usage: dict | None = None
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Calculate cost based on provider pricing (per 1M tokens, as of 2026)."""
        rates = {
            "openai":       {"input": 10.00, "output": 30.00},   # gpt-4o
            "openai-mini":  {"input": 0.15,  "output": 0.60},    # gpt-4o-mini
            "anthropic":    {"input": 3.00,  "output": 15.00},   # claude-sonnet
            "deepseek":     {"input": 0.27,  "output": 1.10},    # deepseek-chat
            "gemini":       {"input": 0.10,  "output": 0.40},    # gemini-2.5-flash
            "ollama":       {"input": 0,     "output": 0},        # local
        }
        # Match closest rate
        rate_key = self.provider
        if self.provider == "openai" and "mini" in self.model:
            rate_key = "openai-mini"
        rate = rates.get(rate_key, {"input": 0, "output": 0})
        return (
            self.total_input_tokens * rate["input"] / 1_000_000
            + self.total_output_tokens * rate["output"] / 1_000_000
        )

    @property
    def last_call_cost_usd(self) -> float:
        """Cost of the most recent API call."""
        if not self._last_usage:
            return 0.0
        rates = {
            "openai":       {"input": 10.00, "output": 30.00},
            "openai-mini":  {"input": 0.15,  "output": 0.60},
            "anthropic":    {"input": 3.00,  "output": 15.00},
            "deepseek":     {"input": 0.27,  "output": 1.10},
            "gemini":       {"input": 0.10,  "output": 0.40},
            "ollama":       {"input": 0,     "output": 0},
        }
        rate_key = self.provider
        if self.provider == "openai" and "mini" in self.model:
            rate_key = "openai-mini"
        rate = rates.get(rate_key, {"input": 0, "output": 0})
        inp = self._last_usage.get("input_tokens", 0)
        out = self._last_usage.get("output_tokens", 0)
        return (inp * rate["input"] + out * rate["output"]) / 1_000_000

    def _record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage from an API call."""
        self._last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "LLMClient":
        """Create a client from the project config file.

        Rules:
        1. Read config.yaml → llm.default_provider
        2. Look up that provider's config in llm.providers.<provider>
        3. Resolve API key from environment variable (${VAR} syntax)
        4. Fall back to provider defaults
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        llm_config = config.get("llm", {})
        provider = llm_config.get("default_provider", "openai")
        providers = llm_config.get("providers", {})

        if provider not in providers:
            raise LLMError(
                f"Default provider '{provider}' not configured in llm.providers"
            )

        provider_config = providers[provider]

        # Resolve API key (may be ${ENV_VAR} reference)
        api_key_raw = provider_config.get("api_key", "")
        api_key = LLMClient._resolve_env(api_key_raw) if api_key_raw else None

        # Resolve base URL for local models
        base_url = provider_config.get("base_url", None)
        if base_url:
            base_url = LLMClient._resolve_env(base_url)

        return cls(
            provider=provider,
            model=provider_config.get("default_model", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
            temperature=provider_config.get("temperature", 0.2),
            max_tokens=provider_config.get("max_tokens", 4096),
            timeout_seconds=provider_config.get("timeout_seconds", 60),
            max_retries=provider_config.get("max_retries", 3),
        )

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve ${ENV_VAR} references in strings."""
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return os.environ.get(var_name, "")
        return value

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Simple chat completion. Returns text response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call(messages)

    def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> dict:
        """Chat completion with JSON output, parsed into a dict.

        The LLM is instructed to return only valid JSON matching the schema.
        Retries on parse failure up to max_retries times.
        """
        schema_json = json.dumps(output_schema, indent=2)
        full_prompt = f"""{system_prompt}

IMPORTANT: You must respond with ONLY valid JSON. No markdown, no code fences, no explanation.

The JSON must follow this schema:
{schema_json}

{user_prompt}"""

        messages = [{"role": "system", "content": full_prompt}]

        for attempt in range(self.max_retries):
            try:
                raw = self._call(messages)
                return self._parse_json(raw, f"Attempt {attempt+1}")
            except ResponseParseError:
                if attempt == self.max_retries - 1:
                    raise
                messages.append({
                    "role": "assistant",
                    "content": raw,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "The response above is not valid JSON. "
                        "Please respond with ONLY valid JSON. No markdown, no explanation."
                    ),
                })
        raise ResponseParseError("Max retries exceeded for JSON parsing")

    def _call(self, messages: list[dict]) -> str:
        """Internal: call the provider. Returns raw text.

        Switches on self.provider to dispatch to the right implementation.
        """
        if not self.api_key and self.provider != "ollama":
            raise APIKeyMissingError(
                f"No API key configured for provider '{self.provider}'. "
                f"Set the key in config.yaml or via environment variable."
            )

        if self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.provider == "deepseek":
            return self._call_openai_compatible(
                messages, base_url="https://api.deepseek.com/v1"
            )
        elif self.provider == "gemini":
            return self._call_openai_compatible(
                messages,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai"
            )
        elif self.provider == "ollama":
            return self._call_ollama(messages)
        else:
            raise LLMError(f"Unknown provider: {self.provider}")

    def _call_openai(self, messages: list[dict]) -> str:
        """Call OpenAI Chat Completion API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                self._record_usage(
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                return content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"OpenAI API error (attempt {attempt+1}): {e}"
                    )
                delay = 2 ** attempt
                time.sleep(delay)

    def _call_openai_compatible(
        self, messages: list[dict], base_url: str
    ) -> str:
        """Call any OpenAI-compatible API (DeepSeek, Gemini, etc.).

        DeepSeek: https://api.deepseek.com/v1
        Gemini:   https://generativelanguage.googleapis.com/v1beta/openai

        Both follow the exact same /chat/completions schema as OpenAI.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=self.api_key, base_url=base_url)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                self._record_usage(
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                return content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"{self.provider} API error (attempt {attempt+1}): {e}"
                    )
                delay = 2 ** attempt
                time.sleep(delay)

    def _call_anthropic(self, messages: list[dict]) -> str:
        """Call Anthropic Messages API."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise LLMError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        client = Anthropic(api_key=self.api_key)

        # Extract system prompt from messages
        system = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                api_messages.append(msg)

        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system or "",
                    messages=api_messages,
                )
                # Claude returns content as blocks
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text = block.text
                        break
                if hasattr(response, "usage") and response.usage:
                    inp = getattr(response.usage, "input_tokens", 0) or 0
                    out = getattr(response.usage, "output_tokens", 0) or 0
                    self._record_usage(inp, out)
                return text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Anthropic API error (attempt {attempt+1}): {e}"
                    )
                delay = 2 ** attempt
                time.sleep(delay)

    def _call_ollama(self, messages: list[dict]) -> str:
        """Call local Ollama server."""
        base_url = self.base_url or "http://localhost:11434"

        try:
            import requests
        except ImportError:
            raise LLMError("requests package not installed")

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Ollama error (attempt {attempt+1}): {e}"
                    )
                delay = 2 ** attempt
                time.sleep(delay)

    @staticmethod
    def _parse_json(raw: str, context: str = "") -> dict:
        """Parse JSON from LLM output, handling common formatting issues.

        Tries multiple strategies:
        1. Raw parse
        2. Strip markdown code fences (```json ... ```)
        3. Extract first { ... } block
        """
        raw = raw.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip ```json``` fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```)
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract first { ... } block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            extracted = raw[start:end+1]
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        raise ResponseParseError(
            f"Failed to parse JSON ({context}). "
            f"Raw output (first 300 chars): {raw[:300]}"
        )
