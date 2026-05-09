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
from retrieval.claim_cache import cache_claims, get_cached_claims
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
    _query_hint = f"{company} {ticker} broker ratings credit ESG mutual fund holdings"
    cached = get_cached_claims(key=ticker, domain="RatingsData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

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
    cache_claims(key=ticker, domain="RatingsData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
