"""
agents/specialist/legal_agent.py
----------------------------------
LegalCheckAgent — SEBI/MCA filings, legal news, risk level assessment.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from tools.legal_tools import check_mca_filings, check_sebi_enforcement, search_legal_news

log = structlog.get_logger(__name__)


def run(company: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    claims: list[SourcedClaim] = []

    for fn, kwargs in [
        (check_sebi_enforcement, {"company": company}),
        (check_mca_filings, {"company": company}),
        (search_legal_news, {"company": company}),
    ]:
        try:
            result = fn.invoke(kwargs)
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            log.warning("legal_step_failed", fn=fn.__name__, error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("legal_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    return claims, elapsed
