"""
agents/specialist/product_agent.py
------------------------------------
ProductAnalysisAgent — product portfolio, industry outlook, market share, segment revenue.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.web_search_tools import (
    analyze_products,
    get_industry_outlook,
    get_market_share,
    get_product_segment_revenue,
)

log = structlog.get_logger(__name__)


def run(company: str, sector: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} {sector} product portfolio market share industry outlook"
    cached = get_cached_claims(key=company, domain="ProductData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (analyze_products, {"company": company}),
        (get_industry_outlook, {"sector": sector}),
        (get_market_share, {"company": company, "sector": sector}),
        (get_product_segment_revenue, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("product_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("product_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=company, domain="ProductData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
