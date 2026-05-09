"""
api/schemas.py
---------------
Pydantic request / response models for the REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=200)
    ticker: str = Field(..., min_length=1, max_length=20)
    market: Literal["US", "INDIA"]
    sector: str = Field(default="General", max_length=100)
    time_horizon_years: Literal[1, 2, 3, 5, 7, 10] = 1

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Analysis job queued"


class AgentStatus(BaseModel):
    agent_name: str
    status: str  # "queued" | "running" | "done" | "failed"
    elapsed_s: Optional[float] = None
    claim_count: Optional[int] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = "queued"
    current_step: str
    completed: bool
    errors: list[str]
    agent_statuses: list[AgentStatus]


class FootnoteOut(BaseModel):
    index: int
    source_name: str
    source_url: str
    fetched_at: datetime


class ForecastProjectionOut(BaseModel):
    year: int
    bull: float
    base: float
    bear: float
    confidence: float


class ForecastReportOut(BaseModel):
    time_horizon_years: int
    overall_stance: str
    confidence: float
    investment_thesis: str
    key_risks: list[str]
    key_catalysts: list[str]
    eps_projections: list[ForecastProjectionOut]
    revenue_projections: list[ForecastProjectionOut]
    price_projections: list[ForecastProjectionOut]


class VerificationSummaryOut(BaseModel):
    total_claims: int
    verified_claims: int
    hallucinated_claims: int
    overall_confidence: float


class ReportResponse(BaseModel):
    job_id: str
    company: str
    ticker: str
    market: str
    time_horizon_years: int
    generated_at: datetime
    markdown_report: str
    footnotes: list[FootnoteOut]
    forecast: ForecastReportOut
    verification: VerificationSummaryOut
    chart_data: dict[str, Any]
    overall_stance: str
    overall_confidence: float


class FeatureImportanceOut(BaseModel):
    feature_name: str
    importance: float
    description: str


class ExplainResponse(BaseModel):
    job_id: str
    agent_name: str
    company: str
    feature_importances: list[FeatureImportanceOut]
    local_prediction: float
    score: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Discovery ("interesting new stocks") schemas
# ---------------------------------------------------------------------------

class DiscoverRequest(BaseModel):
    market: Literal["US", "INDIA"]


class DiscoverJobResponse(BaseModel):
    job_id: str
    market: str
    status: str = "queued"
    message: str = "Discovery job queued"


class StockCandidateOut(BaseModel):
    ticker: str
    company: str
    sector: str
    market: str
    superstar_conviction: float = Field(
        ge=0.0, le=1.0,
        description="0–1: how strongly superstar investors are backing this stock",
    )
    policy_tailwind: float = Field(
        ge=0.0, le=1.0,
        description="0–1: government policy / scheme benefit score",
    )
    financial_score: float = Field(
        ge=0.0, le=1.0,
        description="0–1: quality of fundamentals (higher = better)",
    )
    valuation_score: float = Field(
        ge=0.0, le=1.0,
        description="0–1: attractiveness of current valuation (higher = more undervalued)",
    )
    news_sentiment: float = Field(
        ge=0.0, le=1.0,
        description="0–1: recent news sentiment (higher = more positive)",
    )
    composite_score: float = Field(
        ge=0.0, le=1.0,
        description="Weighted composite score used for final ranking",
    )
    investors_backing: list[str] = Field(
        default_factory=list,
        description="Superstar investors / institutions who recently added this stock",
    )
    policy_catalysts: list[str] = Field(
        default_factory=list,
        description="Government schemes or policies that benefit this stock",
    )
    rationale: str = Field(
        default="",
        description="Brief investment case summary",
    )


class DiscoveryResultOut(BaseModel):
    job_id: str
    market: str
    elapsed_seconds: float
    candidates_evaluated: int
    top_picks: list[StockCandidateOut]
    errors: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
