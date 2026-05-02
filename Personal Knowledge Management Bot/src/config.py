"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root (parent of src/)
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path) if _env_path.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === Required ===
    telegram_bot_token: str = ""

    # === LLM Provider Keys ===
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""

    # === Provider Selection ===
    llm_provider: Literal["deepseek", "openai", "anthropic", "auto"] = "auto"

    # === OpenAI Settings ===
    openai_model: str = "gpt-4o"

    # === Anthropic Settings ===
    anthropic_model: str = "claude-sonnet-4-20250514"

    # === DeepSeek Settings ===
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # === Voice Transcription ===
    transcription_mode: str = "local"
    whisper_model_size: str = "base"

    # === Bot behaviour ===
    max_message_length: int = 4000
    rate_limit_messages: int = 10
    rate_limit_window: int = 60
    worker_count: int = 2

    # === Storage paths ===
    knowledge_base_path: str = "./data/knowledge"
    db_path: str = "./data/bot_state.db"

    # === Logging ===
    log_level: str = "INFO"

    # ---- computed properties ----

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def effective_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.has_deepseek:
            return "deepseek"
        if self.has_openai:
            return "openai"
        if self.has_anthropic:
            return "anthropic"
        return "none"

    @property
    def has_any_llm(self) -> bool:
        return self.has_deepseek or self.has_openai or self.has_anthropic

    @property
    def knowledge_base_dir(self) -> Path:
        return Path(self.knowledge_base_path)

    @property
    def database_path(self) -> Path:
        return Path(self.db_path)


settings = Settings()
