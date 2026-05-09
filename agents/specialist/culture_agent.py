"""
agents/specialist/culture_agent.py
------------------------------------
CultureIntelligenceAgent — Reddit, Twitter, Glassdoor, Ambitionbox, LinkedIn signals.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.culture_tools import (
    aggregate_culture_signal,
    get_ambitionbox_culture,
    get_glassdoor_culture,
    get_new_project_signals,
    get_reddit_employee_sentiment,
    get_twitter_employee_chatter,
)

log = structlog.get_logger(__name__)


def run(company: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    raw_claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (get_reddit_employee_sentiment, {"company": company}),
        (get_twitter_employee_chatter, {"company": company}),
        (get_glassdoor_culture, {"company": company}),
        (get_ambitionbox_culture, {"company": company}),
        (get_new_project_signals, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                raw_claims.extend(result)
        except Exception as exc:
            log.warning("culture_step_failed", fn=fn.__name__, error=str(exc))

    # Synthesise into aggregate culture report
    agg: list[SourcedClaim] = []
    try:
        agg = aggregate_culture_signal.invoke({"claims": raw_claims, "company": company})
    except Exception as exc:
        log.warning("culture_aggregate_failed", error=str(exc))

    claims = raw_claims + agg
    elapsed = time.perf_counter() - t0
    log.info("culture_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
