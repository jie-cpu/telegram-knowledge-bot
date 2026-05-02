"""Global configuration for the ToolAgent system.

All settings are loaded from environment variables with sensible defaults,
so the system can run out of the box without any configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Settings:
    """Application settings, loaded from environment variables with defaults."""

    # ── LLM Configuration ──────────────────────────────────────────
    llm_provider: str = "mock"  # openai, anthropic, deepseek, gemini, mock
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Anthropic-specific
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = ""

    # DeepSeek-specific (OpenAI-compatible API)
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # Gemini-specific
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""

    # ── Agent Loop Configuration ────────────────────────────────────
    max_steps: int = 15
    step_timeout_seconds: int = 60
    session_timeout_seconds: int = 600

    # ── Tool Configuration ──────────────────────────────────────────
    enable_web_search: bool = True
    enable_calculator: bool = True
    enable_database: bool = True
    enable_code_executor: bool = True
    enable_weather: bool = True
    enable_file_manager: bool = True
    enable_send_email: bool = True

    # Code executor sandbox
    code_executor_allowed_modules: str = "math,json,datetime,random,collections,itertools"
    code_executor_timeout: int = 10
    code_executor_max_output_chars: int = 2000

    # ── Guardrails ──────────────────────────────────────────────────
    guardrails_enabled: bool = True
    guardrails_max_input_length: int = 10000
    guardrails_block_pii: bool = True
    guardrails_block_dangerous_code: bool = True
    guardrails_rate_limit_per_minute: int = 30

    # ── Observability ───────────────────────────────────────────────
    observability_enabled: bool = True
    observability_log_dir: str = "logs"
    observability_trace_enabled: bool = True
    observability_metrics_enabled: bool = True

    # ── Demo Mode ───────────────────────────────────────────────────
    demo_mode: bool = False
    demo_data_dir: str = ""


def _load_settings() -> Settings:
    """Load settings from environment variables with defaults."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        max_steps=int(os.getenv("MAX_STEPS", "15")),
        step_timeout_seconds=int(os.getenv("STEP_TIMEOUT_SECONDS", "60")),
        session_timeout_seconds=int(os.getenv("SESSION_TIMEOUT_SECONDS", "600")),
        guardrails_enabled=os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true",
        guardrails_max_input_length=int(os.getenv("GUARDRAILS_MAX_INPUT_LENGTH", "10000")),
        guardrails_rate_limit_per_minute=int(os.getenv("GUARDRAILS_RATE_LIMIT_PER_MINUTE", "30")),
        observability_enabled=os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true",
        observability_log_dir=os.getenv("OBSERVABILITY_LOG_DIR", "logs"),
    )


settings = _load_settings()

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(settings.observability_log_dir)
if not LOG_DIR.is_absolute():
    LOG_DIR = PROJECT_ROOT / LOG_DIR
DEMO_DATA_DIR = Path(settings.demo_data_dir) if settings.demo_data_dir else PROJECT_ROOT / "demo" / "data"
