"""
agents/forecast_agent.py
--------------------------
ForecastSynthesisAgent — applies time-horizon weighting and produces
Bull / Base / Bear projections for EPS, Revenue, Price, and Market Share.
Returns a ForecastReport.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import ForecastReport, SourcedClaim
from tools.forecast_tools import (
    compute_eps_projections,
    compute_market_share_projections,
    compute_price_projections,
    compute_revenue_projections,
    get_time_horizon_weights,
    synthesize_investment_thesis,
)

log = structlog.get_logger(__name__)


def run(
    company: str,
    ticker: str,
    time_horizon_years: int,
    verified_claims: dict[str, list[SourcedClaim]],
) -> tuple[ForecastReport, float]:
    t0 = time.perf_counter()

    financial_claims = verified_claims.get("financial", [])
    technical_claims = verified_claims.get("valuation", [])
    order_book_claims = verified_claims.get("order_book", [])
    market_share_claims = verified_claims.get("product", [])
    innovation_claims = verified_claims.get("innovation", [])
    peer_claims = verified_claims.get("peer", [])
    all_claims_flat = [c for cs in verified_claims.values() for c in cs]

    # Time horizon weights
    weight_claims = get_time_horizon_weights.invoke({"time_horizon_years": time_horizon_years})
    weights: dict = {}
    for c in weight_claims:
        if isinstance(c.value, dict) and "weights" in c.value:
            weights = c.value["weights"]
            break

    # Projections
    eps_claims = compute_eps_projections.invoke({
        "company": company,
        "ticker": ticker,
        "time_horizon_years": time_horizon_years,
        "financial_claims": financial_claims,
        "technical_claims": technical_claims,
        "order_book_claims": order_book_claims,
    })

    revenue_claims = compute_revenue_projections.invoke({
        "company": company,
        "ticker": ticker,
        "time_horizon_years": time_horizon_years,
        "financial_claims": financial_claims,
        "order_book_claims": order_book_claims,
    })

    price_claims = compute_price_projections.invoke({
        "company": company,
        "ticker": ticker,
        "time_horizon_years": time_horizon_years,
        "valuation_claims": technical_claims,
        "technical_claims": technical_claims,
        "eps_projection_claims": eps_claims,
    })

    ms_claims = compute_market_share_projections.invoke({
        "company": company,
        "time_horizon_years": time_horizon_years,
        "market_share_claims": market_share_claims,
        "innovation_claims": innovation_claims,
        "peer_claims": peer_claims,
    })

    # Investment thesis
    thesis_claims = synthesize_investment_thesis.invoke({
        "company": company,
        "ticker": ticker,
        "time_horizon_years": time_horizon_years,
        "all_verified_claims": all_claims_flat[:50],  # cap for LLM context
    })
    thesis_data: dict = {}
    for c in thesis_claims:
        if isinstance(c.value, dict):
            thesis_data = c.value
            break

    # Build YearlyProjection Pydantic objects
    from models.sourced_claim import (
        YearlyEPSProjection,
        YearlyMarketShareProjection,
        YearlyPriceProjection,
        YearlyRevenueProjection,
    )

    def _build_eps(c: SourcedClaim):
        v = c.value
        return YearlyEPSProjection(
            year=int(v.get("year", 0)),
            bull=float(v.get("bull", 0)),
            base=float(v.get("base", 0)),
            bear=float(v.get("bear", 0)),
            growth_driver=str(v.get("growth_driver", "")),
            confidence=float(v.get("confidence", 0.5)),
        )

    def _build_rev(c: SourcedClaim):
        v = c.value
        return YearlyRevenueProjection(
            year=int(v.get("year", 0)),
            bull=float(v.get("bull", 0)),
            base=float(v.get("base", 0)),
            bear=float(v.get("bear", 0)),
            key_driver=str(v.get("key_driver", "")),
            confidence=float(v.get("confidence", 0.5)),
        )

    def _build_price(c: SourcedClaim):
        v = c.value
        return YearlyPriceProjection(
            year=int(v.get("year", 0)),
            bull=float(v.get("bull", 0)),
            base=float(v.get("base", 0)),
            bear=float(v.get("bear", 0)),
            bull_pe=float(v.get("bull_pe", 0)),
            base_pe=float(v.get("base_pe", 0)),
            bear_pe=float(v.get("bear_pe", 0)),
            rationale=str(v.get("rationale", "")),
            confidence=float(v.get("confidence", 0.5)),
        )

    def _build_ms(c: SourcedClaim):
        v = c.value
        return YearlyMarketShareProjection(
            year=int(v.get("year", 0)),
            bull_pct=float(v.get("bull_pct", 0)),
            base_pct=float(v.get("base_pct", 0)),
            bear_pct=float(v.get("bear_pct", 0)),
            catalyst=str(v.get("catalyst", "")),
            confidence=float(v.get("confidence", 0.5)),
        )

    eps_typed = [_build_eps(c) for c in eps_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]
    rev_typed = [_build_rev(c) for c in revenue_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyRevenueProjection"]
    price_typed = [_build_price(c) for c in price_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyPriceProjection"]
    ms_typed = [_build_ms(c) for c in ms_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyMarketShareProjection"]

    projection_years = [p.year for p in eps_typed] or [p.year for p in rev_typed]

    forecast_report = ForecastReport(
        time_horizon_years=time_horizon_years,
        projection_years=projection_years,
        eps_projections=eps_typed,
        revenue_projections=rev_typed,
        price_projections=price_typed,
        market_share_projections=ms_typed,
        dominant_factors=thesis_data.get("dominant_factors", []),
        investment_thesis=thesis_data.get("investment_thesis", ""),
        key_risks=thesis_data.get("key_risks", []),
        key_catalysts=thesis_data.get("key_catalysts", []),
        overall_stance=thesis_data.get("overall_stance", "NEUTRAL"),
        confidence=float(thesis_data.get("confidence", 0.5)),
    )

    elapsed = time.perf_counter() - t0
    log.info("forecast_agent_done", elapsed=round(elapsed, 2), stance=forecast_report.overall_stance)
    return forecast_report, elapsed
