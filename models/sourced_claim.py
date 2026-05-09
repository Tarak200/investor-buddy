"""
models/sourced_claim.py
-----------------------
The atomic data contract for the entire pipeline.
Every data point produced by every tool MUST be wrapped in a SourcedClaim.
Plain strings are forbidden between agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourcedClaim(BaseModel):
    """A single verifiable data point with full source provenance."""

    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    value: Any = Field(..., description="The actual data point or statement.")
    source_url: str = Field(..., description="Exact URL scraped or API endpoint.")
    source_name: str = Field(
        ..., description='Human-readable label, e.g. "Screener.in Balance Sheet".'
    )
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    raw_snippet: str = Field(
        ...,
        max_length=500,
        description="Verbatim text from the source (truncated to 500 chars).",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="0.0–1.0; set by the fetching tool.",
    )
    hallucination_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0.0–1.0; set by VerificationAgent after pipeline.",
    )
    verified: bool = Field(
        default=False,
        description="True once VerificationAgent has evaluated this claim.",
    )

    @property
    def is_disputed(self) -> bool:
        return self.hallucination_score > 0.5

    def truncate_snippet(self) -> None:
        """Ensure raw_snippet does not exceed 500 characters."""
        if len(self.raw_snippet) > 500:
            self.raw_snippet = self.raw_snippet[:497] + "..."


# ---------------------------------------------------------------------------
# Composite output models returned by individual specialist agents
# ---------------------------------------------------------------------------


class OwnershipSnapshot(BaseModel):
    promoter_holding_pct: float = 0.0
    promoter_trend_8q: list[float] = Field(default_factory=list)
    pledge_pct: float = 0.0
    high_pledge_risk: bool = False
    fii_pct: float = 0.0
    dii_pct: float = 0.0
    mf_pct: float = 0.0
    insurance_pct: float = 0.0
    retail_pct: float = 0.0
    top_shareholders: list[dict[str, Any]] = Field(default_factory=list)
    promoter_decreasing: bool = False
    fii_accumulating: bool = False
    fii_trend_slope: float = 0.0
    dii_trend_slope: float = 0.0
    all_claims: list[SourcedClaim] = Field(default_factory=list)


class PeerComparisonReport(BaseModel):
    peers: list[str] = Field(default_factory=list)
    financial_table: list[dict[str, Any]] = Field(default_factory=list)
    price_comparison: dict[str, Any] = Field(default_factory=dict)
    product_price_positioning: dict[str, Any] = Field(default_factory=dict)
    market_share_ranking: dict[str, Any] = Field(default_factory=dict)
    all_claims: list[SourcedClaim] = Field(default_factory=list)


class CultureReport(BaseModel):
    overall_culture_score: float = 0.0
    source_breakdown: dict[str, float] = Field(default_factory=dict)
    dimensions: dict[str, float] = Field(default_factory=dict)
    sentiment_by_department: dict[str, float] = Field(default_factory=dict)
    key_themes_positive: list[str] = Field(default_factory=list)
    key_themes_negative: list[str] = Field(default_factory=list)
    new_project_signals: list[str] = Field(default_factory=list)
    review_volume: dict[str, int] = Field(default_factory=dict)
    recent_trend: str = "STABLE"  # IMPROVING | STABLE | DECLINING
    all_claims: list[SourcedClaim] = Field(default_factory=list)


class InnovationReport(BaseModel):
    rd_spend_abs: list[dict[str, Any]] = Field(default_factory=list)
    rd_intensity_pct: float = 0.0
    rd_vs_sector_avg: float = 0.0
    patent_count_trend: list[dict[str, Any]] = Field(default_factory=list)
    patent_focus_areas: list[str] = Field(default_factory=list)
    global_offices: list[dict[str, Any]] = Field(default_factory=list)
    geographic_revenue_split: dict[str, float] = Field(default_factory=dict)
    international_subsidiaries: list[str] = Field(default_factory=list)
    export_revenue_trend: list[dict[str, Any]] = Field(default_factory=list)
    is_global_player: bool = False
    awards_and_rankings: list[dict[str, Any]] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    all_claims: list[SourcedClaim] = Field(default_factory=list)


class ValuationTechnicalReport(BaseModel):
    # Fundamental valuation
    dcf_intrinsic_value: float = 0.0
    dcf_margin_of_safety_pct: float = 0.0
    pe_vs_sector_pct: float = 0.0
    pb_vs_sector_pct: float = 0.0
    ev_ebitda_vs_sector_pct: float = 0.0
    graham_number: float = 0.0
    peg_ratio: float = 0.0
    valuation_verdict: str = "NEUTRAL"
    valuation_confidence: float = 0.5
    valuation_reasoning: str = ""
    # Technical analysis
    rsi_14: float = 50.0
    macd_signal: str = "NEUTRAL"
    price_vs_200dma: float = 0.0
    golden_cross: bool = False
    death_cross: bool = False
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    volume_trend: str = "NEUTRAL"
    technical_verdict: str = "NEUTRAL"
    technical_reasoning: str = ""
    # New verticals
    new_verticals: list[dict[str, Any]] = Field(default_factory=list)
    total_eps_uplift_3y: float = 0.0
    all_claims: list[SourcedClaim] = Field(default_factory=list)


class ExternalRatingsReport(BaseModel):
    broker_ratings: list[dict[str, Any]] = Field(default_factory=list)
    consensus_rating: str = "HOLD"
    consensus_target_price: float = 0.0
    consensus_upside_pct: float = 0.0
    buy_count: int = 0
    hold_count: int = 0
    sell_count: int = 0
    credit_ratings: list[dict[str, Any]] = Field(default_factory=list)
    rating_trend: str = "STABLE"
    esg_score: float = 0.0
    esg_components: dict[str, float] = Field(default_factory=dict)
    esg_percentile_rank: float = 0.0
    index_memberships: list[str] = Field(default_factory=list)
    recent_index_events: list[dict[str, Any]] = Field(default_factory=list)
    mf_top_holders: list[dict[str, Any]] = Field(default_factory=list)
    mf_net_trend: str = "NEUTRAL"
    government_scheme_benefits: list[dict[str, Any]] = Field(default_factory=list)
    all_claims: list[SourcedClaim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Forecast models
# ---------------------------------------------------------------------------


class YearlyEPSProjection(BaseModel):
    year: int
    bull: float
    base: float
    bear: float
    key_drivers: list[str] = Field(default_factory=list)
    sources: list[SourcedClaim] = Field(default_factory=list)


class YearlyRevenueProjection(BaseModel):
    year: int
    bull_crores: float
    base_crores: float
    bear_crores: float
    revenue_growth_pct_bull: float = 0.0
    revenue_growth_pct_base: float = 0.0
    revenue_growth_pct_bear: float = 0.0
    key_drivers: list[str] = Field(default_factory=list)
    sources: list[SourcedClaim] = Field(default_factory=list)


class YearlyPriceProjection(BaseModel):
    year: int
    bull_target: float
    base_target: float
    bear_target: float
    pe_based: float = 0.0
    dcf_based: float = 0.0
    analyst_extrapolated: float = 0.0
    upside_pct_base: float = 0.0


class YearlyMarketShareProjection(BaseModel):
    year: int
    bull_pct: float
    base_pct: float
    bear_pct: float
    key_assumptions: list[str] = Field(default_factory=list)


class ForecastReport(BaseModel):
    time_horizon_years: int
    projection_years: list[int] = Field(default_factory=list)
    eps_projections: list[YearlyEPSProjection] = Field(default_factory=list)
    revenue_projections: list[YearlyRevenueProjection] = Field(default_factory=list)
    price_projections: list[YearlyPriceProjection] = Field(default_factory=list)
    market_share_projections: list[YearlyMarketShareProjection] = Field(
        default_factory=list
    )
    dominant_factors_for_horizon: list[str] = Field(default_factory=list)
    investment_thesis: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    key_catalysts: list[str] = Field(default_factory=list)
    overall_stance: str = "NEUTRAL"  # BULL | NEUTRAL | BEAR
    overall_stance_reasoning: str = ""
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Final report and LIME
# ---------------------------------------------------------------------------


class VerificationReport(BaseModel):
    per_agent_confidence: dict[str, float] = Field(default_factory=dict)
    disputed_claims: list[SourcedClaim] = Field(default_factory=list)
    overall_pipeline_confidence: float = 0.0
    total_claims: int = 0
    verified_count: int = 0
    hallucinated_count: int = 0
    unverifiable_count: int = 0


class Footnote(BaseModel):
    index: int
    source_name: str
    fetched_at: datetime
    source_url: str
    raw_snippet: str = ""


class FinalReport(BaseModel):
    company: str = ""
    ticker: str
    market: str
    time_horizon_years: int = 1
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str = ""
    body_markdown: str = ""
    markdown_report: str = ""
    overall_stance: str = ""
    footnotes: list[Footnote] = Field(default_factory=list)
    disputed_claims: list[SourcedClaim] = Field(default_factory=list)
    overall_confidence: float = 0.5
    forecast: ForecastReport | None = None
    forecast_report: ForecastReport | None = None
    verification_report: VerificationReport | None = None
    chart_data: dict[str, Any] = Field(default_factory=dict)


class FeatureImportance(BaseModel):
    feature: str
    weight: float
    direction: str  # "positive" | "negative"
    human_label: str = ""


class LIMEExplanation(BaseModel):
    agent_name: str
    feature_importances: list[FeatureImportance] = Field(default_factory=list)
    local_prediction: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    plain_english_summary: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)
