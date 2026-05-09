"""
agents/specialist/superstar_agent.py
--------------------------------------
SuperstarAgent — tracks portfolio additions by influential investors
("superstars") in India and the US, plus institutional new purchases.

India superstars: Vijay Kedia, Ashish Kacholia, Mukul Agarwal, Dolly Khanna,
                  Porinju Veliyath, Ramesh Damani, Raamdeo Agrawal, Nemish Shah

US superstars:    Warren Buffett (Berkshire), Bill Ackman (Pershing Square),
                  Michael Burry (Scion), David Tepper (Appaloosa),
                  Stanley Druckenmiller (Duquesne), Joel Greenblatt (Gotham),
                  Daniel Loeb (Third Point), Seth Klarman (Baupost)

Also scans for large institutional (MF / FII / pension / sovereign wealth)
new additions to portfolios.

Returns: (claims: list[SourcedClaim], elapsed_seconds: float)
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.superstar_tools import (
    get_all_india_superstar_new_picks,
    get_all_us_superstar_new_picks,
    get_institutional_new_additions,
)

log = structlog.get_logger(__name__)


def run(market: str) -> tuple[list[SourcedClaim], float]:
    """
    Execute the SuperstarAgent for the given market.

    Parameters
    ----------
    market : str
        "US" or "INDIA"

    Returns
    -------
    tuple[list[SourcedClaim], float]
        (claims, elapsed_seconds)
    """
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []
    market_upper = market.strip().upper()

    if market_upper == "INDIA":
        steps = [
            ("india_superstars",      lambda: get_all_india_superstar_new_picks.invoke({})),
            ("india_institutional",   lambda: get_institutional_new_additions.invoke({"market": "INDIA"})),
        ]
    else:
        steps = [
            ("us_superstars",         lambda: get_all_us_superstar_new_picks.invoke({})),
            ("us_institutional",      lambda: get_institutional_new_additions.invoke({"market": "US"})),
        ]

    for name, fn in steps:
        try:
            result = fn()
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("superstar_step_failed", step=name, market=market_upper, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info(
        "superstar_agent_done",
        market=market_upper,
        claims=len(claims),
        elapsed=round(elapsed, 2),
    )
    return claims, elapsed
