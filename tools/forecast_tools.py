"""
tools/forecast_tools.py
------------------------
Projection computation and synthesis tools.
Consumes verified agent outputs — makes NO external API calls.

Time-horizon weighting (from plan):
  1Y : Technical 40%, Order Book 25%, Financials 20%, News 10%
  3Y : Financials 35%, Order Book+Verticals 30%, Peers+Share 20%, Mgmt+Culture 10%
  5Y+ : R&D+Innovation 30%, Mgmt+Culture 25%, Market Share+Verticals 25%, Financials 15%
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog
from langchain_core.tools import tool

from config.settings import settings
from llm.provider import get_llm_json_response
from models.sourced_claim import (
    ForecastReport,
    SourcedClaim,
    YearlyEPSProjection,
    YearlyMarketShareProjection,
    YearlyPriceProjection,
    YearlyRevenueProjection,
)
from tools._base import make_claim

log = structlog.get_logger(__name__)

# ── Time-horizon weights ──────────────────────────────────────────────────────

TIME_HORIZON_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "technical": 0.40,
        "order_book": 0.25,
        "financials": 0.20,
        "news": 0.10,
        "other": 0.05,
    },
    2: {
        "technical": 0.35,
        "order_book": 0.25,
        "financials": 0.25,
        "news": 0.10,
        "other": 0.05,
    },
    3: {
        "financials": 0.35,
        "order_book_verticals": 0.30,
        "peers_market_share": 0.20,
        "management_culture": 0.10,
        "other": 0.05,
    },
    5: {
        "rd_innovation": 0.30,
        "management_culture": 0.25,
        "market_share_verticals": 0.25,
        "financials": 0.15,
        "other": 0.05,
    },
    7: {
        "rd_innovation": 0.30,
        "management_culture": 0.25,
        "market_share_verticals": 0.25,
        "financials": 0.15,
        "other": 0.05,
    },
    10: {
        "rd_innovation": 0.30,
        "management_culture": 0.25,
        "market_share_verticals": 0.25,
        "financials": 0.15,
        "other": 0.05,
    },
}


@tool
def get_time_horizon_weights(time_horizon_years: int) -> list[SourcedClaim]:
    """
    Return the weight dictionary for the requested time horizon.
    Returns List[SourcedClaim] containing the weight map.
    """
    # Snap to closest defined horizon
    horizons = sorted(TIME_HORIZON_WEIGHTS.keys())
    chosen = min(horizons, key=lambda h: abs(h - time_horizon_years))
    weights = TIME_HORIZON_WEIGHTS[chosen]

    return [
        make_claim(
            value={"time_horizon_years": time_horizon_years, "weights": weights},
            source_url="internal://forecast_tools",
            source_name="Time Horizon Weights (Plan Spec)",
            raw_snippet=json.dumps(weights)[:500],
            confidence=1.0,
        )
    ]


@tool
def compute_eps_projections(
    company: str,
    ticker: str,
    time_horizon_years: int,
    financial_claims: list[SourcedClaim],
    technical_claims: list[SourcedClaim],
    order_book_claims: list[SourcedClaim],
) -> list[SourcedClaim]:
    """
    Compute Bull / Base / Bear EPS projections for each year in the horizon.
    Returns List[SourcedClaim] each containing a YearlyEPSProjection dict.
    """
    all_data = _extract_text(financial_claims + technical_claims + order_book_claims, 3000)
    current_year = datetime.utcnow().year
    projection_years = list(range(current_year + 1, current_year + time_horizon_years + 1))

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a quantitative equity analyst. "
                "Based on historical financials, technical momentum, and order book data, "
                "generate Bull / Base / Bear EPS projections.\n"
                "Return JSON array (one entry per year):\n"
                '[{"year": int, "bull": float, "base": float, "bear": float, '
                '"growth_driver": str, "confidence": float}]'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}, Ticker: {ticker}\n"
                f"Projection years: {projection_years}\n"
                f"Data:\n{all_data}"
            ),
        },
    ]

    claims: list[SourcedClaim] = []
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        projections: list[dict] = json.loads(response)
        for p in projections:
            claims.append(
                make_claim(
                    value={
                        "type": "YearlyEPSProjection",
                        "year": int(p.get("year", 0)),
                        "bull": float(p.get("bull", 0)),
                        "base": float(p.get("base", 0)),
                        "bear": float(p.get("bear", 0)),
                        "growth_driver": str(p.get("growth_driver", "")),
                        "confidence": float(p.get("confidence", 0.5)),
                        "ticker": ticker,
                    },
                    source_url="internal://forecast_synthesis",
                    source_name="EPS Projection (LLM)",
                    raw_snippet=json.dumps(p)[:500],
                    confidence=float(p.get("confidence", 0.5)),
                )
            )
    except Exception as exc:
        log.error("eps_projection_failed", company=company, error=str(exc))

    return claims


@tool
def compute_revenue_projections(
    company: str,
    ticker: str,
    time_horizon_years: int,
    financial_claims: list[SourcedClaim],
    order_book_claims: list[SourcedClaim],
) -> list[SourcedClaim]:
    """
    Compute Bull / Base / Bear Revenue projections for each year.
    Returns List[SourcedClaim].
    """
    all_data = _extract_text(financial_claims + order_book_claims, 3000)
    current_year = datetime.utcnow().year
    projection_years = list(range(current_year + 1, current_year + time_horizon_years + 1))

    prompt = [
        {
            "role": "system",
            "content": (
                "Generate Bull / Base / Bear Revenue projections (in crores for India, USD millions for US).\n"
                "Return JSON array:\n"
                '[{"year": int, "bull": float, "base": float, "bear": float, '
                '"key_driver": str, "confidence": float}]'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}, Ticker: {ticker}\n"
                f"Projection years: {projection_years}\n"
                f"Data:\n{all_data}"
            ),
        },
    ]

    claims: list[SourcedClaim] = []
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        projections: list[dict] = json.loads(response)
        for p in projections:
            claims.append(
                make_claim(
                    value={
                        "type": "YearlyRevenueProjection",
                        "year": int(p.get("year", 0)),
                        "bull": float(p.get("bull", 0)),
                        "base": float(p.get("base", 0)),
                        "bear": float(p.get("bear", 0)),
                        "key_driver": str(p.get("key_driver", "")),
                        "confidence": float(p.get("confidence", 0.5)),
                        "ticker": ticker,
                    },
                    source_url="internal://forecast_synthesis",
                    source_name="Revenue Projection (LLM)",
                    raw_snippet=json.dumps(p)[:500],
                    confidence=float(p.get("confidence", 0.5)),
                )
            )
    except Exception as exc:
        log.error("revenue_projection_failed", company=company, error=str(exc))

    return claims


@tool
def compute_price_projections(
    company: str,
    ticker: str,
    time_horizon_years: int,
    valuation_claims: list[SourcedClaim],
    technical_claims: list[SourcedClaim],
    eps_projection_claims: list[SourcedClaim],
) -> list[SourcedClaim]:
    """
    Compute Bull / Base / Bear Price projections using P/E × EPS scenario.
    Returns List[SourcedClaim].
    """
    all_data = _extract_text(valuation_claims + technical_claims + eps_projection_claims, 3000)
    current_year = datetime.utcnow().year
    projection_years = list(range(current_year + 1, current_year + time_horizon_years + 1))

    prompt = [
        {
            "role": "system",
            "content": (
                "Generate Bull / Base / Bear Price projections using the provided EPS scenarios "
                "and appropriate P/E multiples for each scenario. "
                "Rationale must mention target P/E multiple and entry/exit prices.\n"
                "Return JSON array:\n"
                '[{"year": int, "bull": float, "base": float, "bear": float, '
                '"bull_pe": float, "base_pe": float, "bear_pe": float, '
                '"rationale": str, "confidence": float}]'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}, Ticker: {ticker}\n"
                f"Projection years: {projection_years}\n"
                f"Data:\n{all_data}"
            ),
        },
    ]

    claims: list[SourcedClaim] = []
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        projections: list[dict] = json.loads(response)
        for p in projections:
            claims.append(
                make_claim(
                    value={
                        "type": "YearlyPriceProjection",
                        "year": int(p.get("year", 0)),
                        "bull": float(p.get("bull", 0)),
                        "base": float(p.get("base", 0)),
                        "bear": float(p.get("bear", 0)),
                        "bull_pe": float(p.get("bull_pe", 0)),
                        "base_pe": float(p.get("base_pe", 0)),
                        "bear_pe": float(p.get("bear_pe", 0)),
                        "rationale": str(p.get("rationale", "")),
                        "confidence": float(p.get("confidence", 0.5)),
                        "ticker": ticker,
                    },
                    source_url="internal://forecast_synthesis",
                    source_name="Price Projection (LLM)",
                    raw_snippet=json.dumps(p)[:500],
                    confidence=float(p.get("confidence", 0.5)),
                )
            )
    except Exception as exc:
        log.error("price_projection_failed", company=company, error=str(exc))

    return claims


@tool
def compute_market_share_projections(
    company: str,
    time_horizon_years: int,
    market_share_claims: list[SourcedClaim],
    innovation_claims: list[SourcedClaim],
    peer_claims: list[SourcedClaim],
) -> list[SourcedClaim]:
    """
    Compute Bull / Base / Bear Market Share % projections.
    Returns List[SourcedClaim].
    """
    all_data = _extract_text(market_share_claims + innovation_claims + peer_claims, 3000)
    current_year = datetime.utcnow().year
    projection_years = list(range(current_year + 1, current_year + time_horizon_years + 1))

    prompt = [
        {
            "role": "system",
            "content": (
                "Estimate Bull / Base / Bear market share percentage for each year. "
                "Consider R&D pipeline, competitive position, and pricing power.\n"
                "Return JSON array:\n"
                '[{"year": int, "bull_pct": float, "base_pct": float, "bear_pct": float, '
                '"catalyst": str, "confidence": float}]'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}\n"
                f"Projection years: {projection_years}\n"
                f"Data:\n{all_data}"
            ),
        },
    ]

    claims: list[SourcedClaim] = []
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        projections: list[dict] = json.loads(response)
        for p in projections:
            claims.append(
                make_claim(
                    value={
                        "type": "YearlyMarketShareProjection",
                        "year": int(p.get("year", 0)),
                        "bull_pct": float(p.get("bull_pct", 0)),
                        "base_pct": float(p.get("base_pct", 0)),
                        "bear_pct": float(p.get("bear_pct", 0)),
                        "catalyst": str(p.get("catalyst", "")),
                        "confidence": float(p.get("confidence", 0.5)),
                        "company": company,
                    },
                    source_url="internal://forecast_synthesis",
                    source_name="Market Share Projection (LLM)",
                    raw_snippet=json.dumps(p)[:500],
                    confidence=float(p.get("confidence", 0.5)),
                )
            )
    except Exception as exc:
        log.error("market_share_projection_failed", company=company, error=str(exc))

    return claims


@tool
def synthesize_investment_thesis(
    company: str,
    ticker: str,
    time_horizon_years: int,
    all_verified_claims: list[SourcedClaim],
) -> list[SourcedClaim]:
    """
    Produce an investment thesis with dominant factors, key risks, catalysts,
    and overall stance (BULL / NEUTRAL / BEAR).
    Returns List[SourcedClaim].
    """
    all_data = _extract_text(all_verified_claims, 4000)
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst. Based on all the verified data provided, "
                "synthesise a concise investment thesis.\n"
                "Return JSON:\n"
                '{"investment_thesis": str, "dominant_factors": [str], '
                '"key_risks": [str], "key_catalysts": [str], '
                '"overall_stance": "BULL" | "NEUTRAL" | "BEAR", '
                '"confidence": float}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}, Ticker: {ticker}, Time Horizon: {time_horizon_years}Y\n"
                f"Data:\n{all_data}"
            ),
        },
    ]

    claims: list[SourcedClaim] = []
    try:
        response = get_llm_json_response(prompt, temperature=0.1, max_tokens=1024)
        parsed = json.loads(response)
        claims.append(
            make_claim(
                value={**parsed, "company": company, "ticker": ticker, "time_horizon_years": time_horizon_years},
                source_url="internal://forecast_synthesis",
                source_name="Investment Thesis (LLM)",
                raw_snippet=str(parsed.get("investment_thesis", ""))[:500],
                confidence=float(parsed.get("confidence", 0.6)),
            )
        )
    except Exception as exc:
        log.error("investment_thesis_failed", company=company, error=str(exc))

    return claims


# ── Internal helper ────────────────────────────────────────────────────────────


def _extract_text(claims: list[SourcedClaim], max_chars: int = 4000) -> str:
    """Flatten claim values to a truncated string for LLM prompts."""
    parts: list[str] = []
    for c in claims:
        if isinstance(c.value, dict):
            parts.append(json.dumps(c.value, default=str))
        else:
            parts.append(str(c.value))
    full = " ".join(parts)
    return full[:max_chars]
