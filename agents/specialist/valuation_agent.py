"""
agents/specialist/valuation_agent.py
--------------------------------------
ValuationTechnicalAgent — DCF, relative valuation, Graham Number, PEG,
technicals, new verticals, and EPS impact.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.valuation_tools import (
    compute_valuation_verdict,
    estimate_vertical_eps_impact,
    get_graham_number,
    get_intrinsic_value_dcf,
    get_new_verticals,
    get_peg_ratio,
    get_price_technicals,
    get_relative_valuation,
    get_technical_verdict,
)

log = structlog.get_logger(__name__)


def run(
    company: str,
    ticker: str,
    market: str,
    peer_list: list[str] | None = None,
) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []
    peers = peer_list or []

    for fn, kwargs in [
        (get_intrinsic_value_dcf, {"ticker": ticker, "market": market}),
        (get_relative_valuation, {"ticker": ticker, "market": market, "peer_list": peers}),
        (get_graham_number, {"ticker": ticker}),
        (get_peg_ratio, {"ticker": ticker}),
        (get_price_technicals, {"ticker": ticker, "market": market}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("valuation_step_failed", fn=fn.__name__, error=str(exc))

    # Aggregate verdicts
    try:
        val_verdict = compute_valuation_verdict.invoke({"claims": claims})
        if isinstance(val_verdict, list):
            claims.extend(val_verdict)
    except Exception as exc:
        log.warning("valuation_verdict_failed", error=str(exc))

    try:
        tech_verdict = get_technical_verdict.invoke({"technical_claims": claims})
        if isinstance(tech_verdict, list):
            claims.extend(tech_verdict)
    except Exception as exc:
        log.warning("technical_verdict_failed", error=str(exc))

    # New verticals
    vertical_claims: list[SourcedClaim] = []
    try:
        vertical_claims = get_new_verticals.invoke({"company": company})
        if isinstance(vertical_claims, list):
            claims.extend(vertical_claims)
    except Exception as exc:
        log.warning("new_verticals_failed", error=str(exc))

    try:
        eps_impact = estimate_vertical_eps_impact.invoke({"company": company, "verticals": vertical_claims})
        if isinstance(eps_impact, list):
            claims.extend(eps_impact)
    except Exception as exc:
        log.warning("vertical_eps_impact_failed", error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("valuation_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
