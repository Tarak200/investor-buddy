"""
agents/specialist/ownership_agent.py
--------------------------------------
OwnershipAgent — promoter holding, pledge risk, institutional ownership, shareholder trends.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.ownership_tools import (
    get_institutional_ownership,
    get_ownership_trend,
    get_promoter_holding,
    get_promoter_pledging,
    get_top_shareholders,
)

log = structlog.get_logger(__name__)


def run(company: str, ticker: str, market: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (get_promoter_holding, {"ticker": ticker, "market": market}),
        (get_promoter_pledging, {"ticker": ticker}),
        (get_institutional_ownership, {"ticker": ticker, "market": market}),
        (get_top_shareholders, {"ticker": ticker}),
        (get_ownership_trend, {"ticker": ticker, "market": market}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("ownership_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("ownership_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
