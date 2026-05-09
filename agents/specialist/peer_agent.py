"""
agents/specialist/peer_agent.py
---------------------------------
PeerComparisonAgent — peer list, financials, price comparison, market share benchmarking.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.peer_tools import (
    get_peer_financials,
    get_peer_list,
    get_peer_market_share_comparison,
    get_price_comparison,
    get_product_price_benchmarking,
)
from tools.web_search_tools import get_market_share

log = structlog.get_logger(__name__)


def run(company: str, ticker: str, market: str, sector: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} {ticker} {sector} peer comparison benchmarking"
    cached = get_cached_claims(key=ticker, domain="PeerData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []

    # Step 1: peer list
    peer_list: list[str] = []
    try:
        peer_claims = get_peer_list.invoke({"company": company, "sector": sector})
        if isinstance(peer_claims, list):
            claims.extend(peer_claims)
            for c in peer_claims:
                if isinstance(c.value, dict) and "peer_list" in c.value:
                    peer_list = c.value["peer_list"][:8]
                    break
    except Exception as exc:
        log.warning("peer_list_failed", error=str(exc))

    # Remaining steps using peer list
    for fn, kwargs in [
        (get_peer_financials, {"peer_list": peer_list}),
        (get_price_comparison, {"company_ticker": ticker, "peer_list": peer_list}),
        (get_peer_market_share_comparison, {"company": company, "sector": sector}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("peer_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("peer_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=ticker, domain="PeerData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
