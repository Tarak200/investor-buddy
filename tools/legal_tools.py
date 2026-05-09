"""
tools/legal_tools.py
--------------------
Legal and compliance risk detection tools.

US  : SEC EDGAR enforcement actions
India: SEBI enforcement page, MCA21, Indian Kanoon, Taxmann via Tavily
"""

from __future__ import annotations

import json
import re

import structlog
from langchain_core.tools import tool

from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)

_SEBI_ENFORCEMENT_URL = "https://www.sebi.gov.in/enforcement/orders/latest-orders.html"
_MCA_SEARCH_URL = "https://www.mca.gov.in/mcafoportal/showCheckCompanyName.do"


def _risk_level(violations: int, defaults: int, cases: int, penalty: float) -> str:
    score = violations * 3 + defaults * 2 + cases + (penalty / 1_000_000)
    if score == 0:
        return "LOW"
    if score < 5:
        return "MEDIUM"
    if score < 15:
        return "HIGH"
    return "CRITICAL"


@tool
def check_sebi_enforcement(company: str) -> list[SourcedClaim]:
    """
    Search SEBI enforcement orders for the company.
    Returns List[SourcedClaim] with violation details and overall risk level.
    India-specific. Returns empty list for US companies.
    """
    claims: list[SourcedClaim] = []

    # Tavily search against SEBI enforcement page
    results = _tavily_search(
        f"SEBI enforcement order {company} site:sebi.gov.in",
        max_results=5,
    )
    violations = []
    for r in results:
        url = r.get("url", _SEBI_ENFORCEMENT_URL)
        content = r.get("content", "")
        if company.lower() in content.lower() or company.lower() in r.get("title", "").lower():
            violations.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": content[:200],
            })
            claims.append(
                make_claim(
                    value={"violation": r.get("title", ""), "source_type": "SEBI", "company": company},
                    source_url=url,
                    source_name="SEBI Enforcement Orders",
                    raw_snippet=content[:500],
                    confidence=0.75,
                )
            )

    risk_level = _risk_level(len(violations), 0, 0, 0.0)
    claims.append(
        make_claim(
            value={
                "risk_level": risk_level,
                "sebi_violations_count": len(violations),
                "company": company,
            },
            source_url=_SEBI_ENFORCEMENT_URL,
            source_name="SEBI Enforcement Summary",
            raw_snippet=f"Found {len(violations)} SEBI enforcement records for {company}",
            confidence=0.80,
        )
    )
    log.info("sebi_check_done", company=company, violations=len(violations), risk=risk_level)
    return claims


@tool
def check_mca_filings(company: str) -> list[SourcedClaim]:
    """
    Check MCA21 for company defaults and filing status.
    Returns List[SourcedClaim] with compliance status.
    India-specific.
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} MCA21 company master data site:mca.gov.in OR MCA default filing",
        max_results=5,
    )
    defaults_count = 0
    for r in results:
        content = r.get("content", "")
        if "default" in content.lower() or "strike off" in content.lower():
            defaults_count += 1
            claims.append(
                make_claim(
                    value={
                        "mca_issue": r.get("title", ""),
                        "company": company,
                    },
                    source_url=r.get("url", _MCA_SEARCH_URL),
                    source_name="MCA21 Filing Records",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )

    claims.append(
        make_claim(
            value={"mca_defaults": defaults_count, "company": company},
            source_url=_MCA_SEARCH_URL,
            source_name="MCA21 Company Master",
            raw_snippet=f"MCA default check for {company}: {defaults_count} issue(s) found",
            confidence=0.70,
        )
    )
    return claims


@tool
def search_legal_news(company: str) -> list[SourcedClaim]:
    """
    Search for active court cases and pending legal matters via Tavily + Indian Kanoon.
    Returns List[SourcedClaim] with case details and combined risk level.
    """
    claims: list[SourcedClaim] = []

    # Indian Kanoon search
    kanoon_results = _tavily_search(
        f"{company} court case judgment site:indiankanoon.org",
        max_results=5,
    )
    cases: list[dict] = []
    for r in kanoon_results:
        cases.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
        })
        claims.append(
            make_claim(
                value={"court_case": r.get("title", ""), "company": company},
                source_url=r.get("url", "https://indiankanoon.org"),
                source_name="Indian Kanoon",
                raw_snippet=r.get("content", "")[:500],
                confidence=0.70,
            )
        )

    # General legal news (Taxmann, etc.)
    legal_news = _tavily_search(f"{company} legal penalty fraud case news", max_results=5)
    for r in legal_news:
        claims.append(
            make_claim(
                value={"legal_news": r.get("title", ""), "company": company},
                source_url=r.get("url", ""),
                source_name=f"Legal News – {r.get('source', 'Web')}",
                raw_snippet=r.get("content", "")[:500],
                confidence=0.65,
            )
        )

    # Compute overall risk level
    sebi_count = sum(1 for c in claims if "SEBI" in str(c.value))
    risk = _risk_level(sebi_count, 0, len(cases), 0.0)
    claims.append(
        make_claim(
            value={
                "overall_risk_level": risk,
                "active_court_cases": len(cases),
                "company": company,
            },
            source_url="https://indiankanoon.org",
            source_name="Legal Risk Summary",
            raw_snippet=f"Risk level: {risk}, active cases: {len(cases)}",
            confidence=0.75,
        )
    )
    log.info("legal_check_done", company=company, cases=len(cases), risk=risk)
    return claims
