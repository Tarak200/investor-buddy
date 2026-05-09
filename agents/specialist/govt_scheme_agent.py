"""
agents/specialist/govt_scheme_agent.py
----------------------------------------
GovtSchemeAgent — scans government policies, regulatory notifications, budget schemes,
and sector incentives / penalties that can create investment opportunities or risks.

India coverage: Union Budget, PLI schemes, SEBI circulars, RBI policy,
                Ministry of Finance / Commerce / Industry notifications,
                infrastructure mandates, green energy missions.

US coverage:    Inflation Reduction Act (IRA), CHIPS and Science Act,
                Federal Reserve rate policy, SEC regulatory changes,
                Executive Orders, trade tariffs, congressional acts.

Returns: (claims: list[SourcedClaim], elapsed_seconds: float)
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.govt_scheme_tools import (
    get_india_budget_highlights,
    get_india_govt_schemes,
    get_policy_driven_stock_ideas,
    get_us_govt_policies,
    get_us_ira_chips_beneficiaries,
)

log = structlog.get_logger(__name__)


def run(market: str) -> tuple[list[SourcedClaim], float]:
    """
    Execute the GovtSchemeAgent for the given market.

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
            ("india_budget",       lambda: get_india_budget_highlights.invoke({})),
            ("india_schemes",      lambda: get_india_govt_schemes.invoke({"sector": ""})),
            ("india_policy_picks", lambda: get_policy_driven_stock_ideas.invoke({"market": "INDIA"})),
        ]
    else:
        steps = [
            ("us_policies",        lambda: get_us_govt_policies.invoke({"sector": ""})),
            ("us_ira_chips",       lambda: get_us_ira_chips_beneficiaries.invoke({})),
            ("us_policy_picks",    lambda: get_policy_driven_stock_ideas.invoke({"market": "US"})),
        ]

    for name, fn in steps:
        try:
            result = fn()
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning(
                "govt_scheme_step_failed",
                step=name,
                market=market_upper,
                error=str(exc),
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "govt_scheme_agent_done",
        market=market_upper,
        claims=len(claims),
        elapsed=round(elapsed, 2),
    )
    return claims, elapsed
