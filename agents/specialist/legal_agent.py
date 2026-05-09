"""
agents/specialist/legal_agent.py
----------------------------------
LegalCheckAgent — SEBI/MCA filings, legal news, risk level assessment.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.legal_tools import check_mca_filings, check_sebi_enforcement, search_legal_news

log = structlog.get_logger(__name__)


def run(company: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} legal SEBI MCA filings enforcement"
    cached = get_cached_claims(key=company, domain="LegalRecords", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

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
    cache_claims(key=company, domain="LegalRecords", claims=claims, query_hint=_query_hint)
    return claims, elapsed
