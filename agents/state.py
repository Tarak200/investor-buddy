"""
agents/state.py
----------------
Shared LangGraph state TypedDict for the entire analysis pipeline.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, NotRequired

from models.sourced_claim import (
    FinalReport,
    ForecastReport,
    LIMEExplanation,
    SourcedClaim,
    VerificationReport,
)


class AnalysisState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    company: str
    ticker: str
    market: str                          # "US" | "INDIA"
    sector: str
    time_horizon_years: int              # 1 | 2 | 3 | 5 | 7 | 10
    job_id: str

    # ── Parallel gather outputs (12 specialist agents) ─────────────────────
    financial_claims: NotRequired[list[SourcedClaim]]
    news_claims: NotRequired[list[SourcedClaim]]
    legal_claims: NotRequired[list[SourcedClaim]]
    order_book_claims: NotRequired[list[SourcedClaim]]
    product_claims: NotRequired[list[SourcedClaim]]
    management_claims: NotRequired[list[SourcedClaim]]
    ownership_claims: NotRequired[list[SourcedClaim]]
    peer_claims: NotRequired[list[SourcedClaim]]
    culture_claims: NotRequired[list[SourcedClaim]]
    innovation_claims: NotRequired[list[SourcedClaim]]
    valuation_claims: NotRequired[list[SourcedClaim]]
    ratings_claims: NotRequired[list[SourcedClaim]]

    # ── Orchestration outputs ─────────────────────────────────────────────
    verification_report: NotRequired[VerificationReport]
    forecast_report: NotRequired[ForecastReport]
    chart_data: NotRequired[dict[str, Any]]
    final_report: NotRequired[FinalReport]

    # ── Per-agent latency tracking ─────────────────────────────────────────
    agent_latencies: NotRequired[dict[str, float]]

    # ── LIME (on-demand only) ─────────────────────────────────────────────
    lime_explanation: NotRequired[Optional[LIMEExplanation]]

    # ── Pipeline control ──────────────────────────────────────────────────
    errors: NotRequired[list[str]]
    current_step: NotRequired[str]
