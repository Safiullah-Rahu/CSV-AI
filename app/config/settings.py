"""Application settings backed by environment variables.

Uses pydantic-settings so values can come from `.env`, real environment, or
Streamlit secrets. All UI/services should pull config from here rather than
reading ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal["openai", "anthropic", "ollama"]


class Settings(BaseSettings):
    """Runtime settings for CSV-AI.

    Loaded from environment variables and/or a ``.env`` file. Streamlit
    secrets are merged in by :func:`get_settings` so the same code works on
    Streamlit Community Cloud and locally.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    app_name: str = "CSV-AI"
    environment: Literal["dev", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- LLM provider ------------------------------------------------------
    default_provider: ProviderName = "openai"
    default_model: str = "gpt-4o-mini"
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1024, ge=64, le=32000)

    # --- Provider credentials ---------------------------------------------
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # --- Data handling ----------------------------------------------------
    max_csv_rows_in_context: int = 200
    sample_rows_for_preview: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- UI ----------------------------------------------------------------
    show_token_usage: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    On Streamlit Community Cloud, ``st.secrets`` provides API keys. We try
    to merge them in (best-effort — failures fall back to env-only).
    """

    settings = Settings()

    try:  # pragma: no cover - exercised only inside Streamlit
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets"):
            secrets = dict(st.secrets)  # may raise if no secrets.toml
            for key in ("openai_api_key", "anthropic_api_key", "ollama_base_url"):
                value = secrets.get(key) or secrets.get(key.upper())
                if value and not getattr(settings, key, None):
                    object.__setattr__(settings, key, value)
    except Exception:
        # No Streamlit context, or no secrets file. Safe to ignore.
        pass

    return settings
