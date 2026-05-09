"""
config/settings.py
------------------
All application settings loaded from environment variables / .env file.
Every hard-coded constant in the codebase should live here.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM API keys
    # ------------------------------------------------------------------
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # ------------------------------------------------------------------
    # LLM model selection
    # ------------------------------------------------------------------
    llm_primary_model: str = Field(
        default="groq/llama-3.3-70b-versatile",
        alias="LLM_PRIMARY_MODEL",
    )
    llm_fallback_model_1: str = Field(
        default="groq/llama-3.1-8b-instant",
        alias="LLM_FALLBACK_MODEL_1",
    )
    llm_fallback_model_2: str = Field(
        default="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        alias="LLM_FALLBACK_MODEL_2",
    )
    llm_fallback_model_3: str = Field(
        default="gemini/gemini-1.5-flash",
        alias="LLM_FALLBACK_MODEL_3",
    )

    # ------------------------------------------------------------------
    # Data API keys
    # ------------------------------------------------------------------
    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(
        default="financial_assistant/0.1",
        alias="REDDIT_USER_AGENT",
    )

    # ------------------------------------------------------------------
    # Time horizons (data fetching windows)
    # ------------------------------------------------------------------
    news_lookback_days: int = Field(default=30, alias="NEWS_LOOKBACK_DAYS")
    technical_lookback_days: int = Field(default=180, alias="TECHNICAL_LOOKBACK_DAYS")
    financials_lookback_years: int = Field(default=5, alias="FINANCIALS_LOOKBACK_YEARS")
    eps_history_years: int = Field(default=10, alias="EPS_HISTORY_YEARS")
    quarterly_results_count: int = Field(default=8, alias="QUARTERLY_RESULTS_COUNT")
    promoter_holding_quarters: int = Field(default=8, alias="PROMOTER_HOLDING_QUARTERS")

    # ------------------------------------------------------------------
    # Cache TTLs (seconds)
    # ------------------------------------------------------------------
    cache_ttl_financials: int = Field(default=86400, alias="CACHE_TTL_FINANCIALS")
    cache_ttl_news: int = Field(default=3600, alias="CACHE_TTL_NEWS")

    # ------------------------------------------------------------------
    # Forward projection horizons (comma-separated years)
    # ------------------------------------------------------------------
    projection_horizons: str = Field(default="1,3,5", alias="PROJECTION_HORIZONS")

    @property
    def projection_horizon_list(self) -> list[int]:
        return [int(h.strip()) for h in self.projection_horizons.split(",")]

    # ------------------------------------------------------------------
    # Weaviate vector store
    # ------------------------------------------------------------------
    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    weaviate_api_key: str = Field(default="", alias="WEAVIATE_API_KEY")

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ------------------------------------------------------------------
    # LangSmith (optional evaluation tracing)
    # ------------------------------------------------------------------
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(
        default="financial-assistant",
        alias="LANGCHAIN_PROJECT",
    )

    # ------------------------------------------------------------------
    # AWS (production deployment only)
    # ------------------------------------------------------------------
    aws_region: str = Field(default="ap-south-1", alias="AWS_REGION")
    aws_s3_bucket: str = Field(default="", alias="AWS_S3_BUCKET")

    # ------------------------------------------------------------------
    # Hallucination thresholds
    # ------------------------------------------------------------------
    hallucination_threshold: float = Field(
        default=0.7,
        alias="HALLUCINATION_THRESHOLD",
    )
    min_confidence_for_report: float = Field(
        default=0.5,
        alias="MIN_CONFIDENCE_FOR_REPORT",
    )

    # ------------------------------------------------------------------
    # Pledge risk threshold (%)
    # ------------------------------------------------------------------
    pledge_high_risk_threshold: float = Field(
        default=20.0,
        alias="PLEDGE_HIGH_RISK_THRESHOLD",
    )

    # ------------------------------------------------------------------
    # Global player threshold (% export revenue)
    # ------------------------------------------------------------------
    global_player_export_threshold: float = Field(
        default=30.0,
        alias="GLOBAL_PLAYER_EXPORT_THRESHOLD",
    )

    # ------------------------------------------------------------------
    # Discovery pipeline tuning
    # ------------------------------------------------------------------
    discovery_max_candidates: int = Field(
        default=10,
        alias="DISCOVERY_MAX_CANDIDATES",
    )
    discovery_top_picks: int = Field(
        default=5,
        alias="DISCOVERY_TOP_PICKS",
    )
    discovery_deep_dive_workers: int = Field(
        default=3,
        alias="DISCOVERY_DEEP_DIVE_WORKERS",
    )


# Singleton – import this everywhere
settings = Settings()
