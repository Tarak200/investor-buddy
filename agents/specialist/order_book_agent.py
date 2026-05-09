"""
agents/specialist/order_book_agent.py
--------------------------------------
OrderBookAgent — government tenders + order book data.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.tender_tools import get_order_book_data, search_government_tenders

log = structlog.get_logger(__name__)


def run(company: str, market: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
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
    return claims, elapsed
