"""
agents/specialist/ratings_agent.py
------------------------------------
ExternalRatingsAgent — broker recommendations, credit ratings, ESG, index memberships,
MF holdings, and government scheme benefits.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.ratings_tools import (
    get_broker_recommendations,
    get_credit_ratings,
    get_esg_scores,
    get_government_schemes_benefit,
    get_index_memberships,
    get_mutual_fund_holdings,
)

log = structlog.get_logger(__name__)


def run(company: str, ticker: str, market: str, sector: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (get_broker_recommendations, {"company": company, "ticker": ticker, "market": market}),
        (get_credit_ratings, {"company": company, "market": market}),
        (get_esg_scores, {"company": company}),
        (get_index_memberships, {"ticker": ticker, "market": market}),
        (get_mutual_fund_holdings, {"ticker": ticker, "market": market}),
        (get_government_schemes_benefit, {"company": company, "sector": sector}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("ratings_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("ratings_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
