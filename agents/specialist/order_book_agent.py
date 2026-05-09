"""
agents/specialist/order_book_agent.py
--------------------------------------
OrderBookAgent — government tenders + order book data.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.tender_tools import get_order_book_data, search_government_tenders

log = structlog.get_logger(__name__)


def run(company: str, market: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} government tenders order book"
    cached = get_cached_claims(key=company, domain="TenderData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (search_government_tenders, {"company": company, "market": market}),
        (get_order_book_data, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("order_book_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("order_book_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=company, domain="TenderData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
