"""Platform settings backed by environment variables.

Originally the CSV-AI Streamlit settings; now the single source of truth for the
Enterprise RAG platform. Values come from the process environment or a ``.env``
file. Nothing in the code base should read ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["openai", "anthropic", "ollama"]
CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    app_name: str = "Enterprise RAG & Knowledge Intelligence Platform"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    public_app_url: str = "http://localhost:3000"
    api_prefix: str = "/api/v1"
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Database ----------------------------------------------------------
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_statement_timeout_ms: int = 30_000

    # --- Security ----------------------------------------------------------
    # JWT signing secret. MUST be overridden outside dev/test (checked at startup).
    jwt_secret: str = "dev-only-change-me-dev-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 14
    refresh_cookie_name: str = "rag_refresh"
    cookie_secure: bool = False
    # Fernet keys (urlsafe base64, 32 bytes). First key encrypts; all keys decrypt (rotation).
    encryption_keys: CsvList = Field(default_factory=list)
    password_min_length: int = 10
    email_verification_required: bool = False
    login_rate_limit_per_minute: int = 10
    register_rate_limit_per_hour: int = 10
    api_rate_limit_per_minute: int = 600
    # Allow database connections / webhooks to private network ranges (self-hosted default).
    allow_private_network_targets: bool = True
    # Website / sitemap / RSS ingestion may reach private hosts only when explicitly allowed.
    allow_private_web_ingestion: bool = False
    ssrf_allowlist: CsvList = Field(default_factory=list)
    # SQLite connections may only open files inside these directories (read-only).
    sqlite_allowed_dirs: CsvList = Field(default_factory=lambda: ["./var/sqlite"])

    # --- Storage -----------------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: str = "./var/storage"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "rag-documents"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    max_upload_mb: int = 100
    max_batch_files: int = 50

    # --- Jobs / events -----------------------------------------------------
    # embedded: worker threads inside the API process (single node / dev)
    # external: API only enqueues; `python -m app.jobs.worker` processes execute
    # eager:    run synchronously right after the enqueuing transaction commits (tests)
    job_backend: Literal["embedded", "external", "eager"] = "embedded"
    job_max_attempts: int = 4
    job_backoff_base_seconds: float = 2.0
    worker_concurrency: int = 4
    job_lease_minutes: int = 30
    redis_url: str | None = None
    event_backend: Literal["memory", "redis"] = "memory"
    scheduler_enabled: bool = True
    scheduler_interval_seconds: int = 30

    # --- Email ---------------------------------------------------------------
    email_backend: Literal["console", "smtp", "memory"] = "console"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "no-reply@localhost"
    smtp_starttls: bool = True

    # --- AI defaults (platform level; organisations configure their own) --------
    default_provider: ProviderName = "openai"
    default_model: str = "gpt-4o-mini"
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1024, ge=64, le=32000)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    anthropic_workspace_id: str | None = None  # needed by organisation-level Anthropic keys
    ollama_base_url: str = "http://localhost:11434"
    default_embedding_provider: str = "hashing"
    default_embedding_model: str = "hashing-768"
    llm_timeout_seconds: float = 120.0
    # Use the platform's ANTHROPIC_API_KEY / OPENAI_API_KEY for organisations without their own
    # provider (self-hosted only). DEFAULT_PROVIDER is tried first; DEFAULT_MODEL applies when it
    # belongs to that provider, otherwise the provider's recommended model is used.
    platform_llm_fallback: bool = False

    # --- CSV analysis (retained from CSV-AI) ------------------------------------
    max_csv_rows_in_context: int = 200
    sample_rows_for_preview: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- OAuth apps for SaaS integrations (platform-level; redirect URI:
    # {PUBLIC_APP_URL}/api/v1/integrations/oauth/callback). Unset = that vendor's
    # OAuth connection is shown as "needs setup".
    oauth_google_client_id: str | None = None
    oauth_google_client_secret: str | None = None
    oauth_microsoft_client_id: str | None = None
    oauth_microsoft_client_secret: str | None = None
    oauth_microsoft_tenant: str = "common"
    oauth_dropbox_client_id: str | None = None
    oauth_dropbox_client_secret: str | None = None
    oauth_notion_client_id: str | None = None
    oauth_notion_client_secret: str | None = None
    oauth_atlassian_client_id: str | None = None
    oauth_atlassian_client_secret: str | None = None
    oauth_slack_client_id: str | None = None
    oauth_slack_client_secret: str | None = None

    # --- Billing -------------------------------------------------------------
    billing_enabled: bool = False
    default_plan: str = "enterprise"
    stripe_secret_key: str | None = None

    # --- Bootstrap -------------------------------------------------------------
    bootstrap_admin_email: str | None = None
    bootstrap_admin_password: str | None = None

    @field_validator("cors_origins", "encryption_keys", "ssrf_allowlist", "sqlite_allowed_dirs", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    def validate_for_production(self) -> list[str]:
        """Return a list of fatal misconfigurations for a production start."""
        problems: list[str] = []
        if not self.is_production:
            return problems
        if "dev-only" in self.jwt_secret or len(self.jwt_secret) < 32:
            problems.append("JWT_SECRET must be a random value of at least 32 characters.")
        if not self.encryption_keys:
            problems.append("ENCRYPTION_KEYS must contain at least one Fernet key.")
        if not self.cookie_secure:
            problems.append("COOKIE_SECURE must be true in production (HTTPS only).")
        if self.email_backend == "console":
            problems.append("EMAIL_BACKEND=console is not allowed in production.")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
