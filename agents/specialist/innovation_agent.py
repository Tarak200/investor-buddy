"""
agents/specialist/innovation_agent.py
---------------------------------------
InnovationGlobalAgent — R&D, patents, global presence, export, certifications.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.innovation_tools import (
    get_certifications,
    get_competition_rankings,
    get_export_trends,
    get_geographic_revenue_split,
    get_global_presence,
    get_international_subsidiaries,
    get_patent_activity,
    get_rd_spending,
)

log = structlog.get_logger(__name__)


def run(company: str, ticker: str, market: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} {ticker} R&D patents innovation global presence exports"
    cached = get_cached_claims(key=ticker, domain="InnovationData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (get_rd_spending, {"company": company, "ticker": ticker, "market": market}),
        (get_patent_activity, {"company": company}),
        (get_global_presence, {"company": company}),
        (get_geographic_revenue_split, {"company": company}),
        (get_international_subsidiaries, {"company": company}),
        (get_export_trends, {"company": company}),
        (get_competition_rankings, {"company": company, "products": []}),
        (get_certifications, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("innovation_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("innovation_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=ticker, domain="InnovationData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
