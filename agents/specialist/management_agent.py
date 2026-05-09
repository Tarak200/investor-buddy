"""
agents/specialist/management_agent.py
---------------------------------------
ManagementAgent — executive profiles, employee reviews, board qualifications.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.management_tools import get_board_qualifications, get_employee_reviews, get_management_profiles

log = structlog.get_logger(__name__)


def run(company: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (get_management_profiles, {"company": company}),
        (get_employee_reviews, {"company": company}),
        (get_board_qualifications, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("management_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("management_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
