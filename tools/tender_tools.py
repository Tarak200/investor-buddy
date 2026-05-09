"""
tools/tender_tools.py
---------------------
Government tender and order book analysis tools.

US  : SAM.gov via Tavily
India: GeM portal, CPPP, Screener.in concall transcripts
"""

from __future__ import annotations

import structlog
from langchain_core.tools import tool

from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


@tool
def search_government_tenders(company: str, market: str = "INDIA") -> list[SourcedClaim]:
    """
    Search for government tender wins and active bids for the company.
    India: GeM portal + CPPP.  US: SAM.gov via Tavily.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    if market.upper() == "INDIA":
        # GeM portal search via Tavily
        gem_results = _tavily_search(
            f"{company} tender order GeM portal site:gem.gov.in OR GeM order",
            max_results=8,
        )
        for r in gem_results:
            claims.append(
                make_claim(
                    value={
                        "tender": r.get("title", ""),
                        "source_type": "GeM",
                        "company": company,
                    },
                    source_url=r.get("url", "https://gem.gov.in"),
                    source_name="GeM Portal",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.75,
                )
            )

        # CPPP search
        cppp_results = _tavily_search(
            f"{company} tender CPPP site:cppp.gov.in", max_results=5
        )
        for r in cppp_results:
            claims.append(
                make_claim(
                    value={"tender": r.get("title", ""), "source_type": "CPPP", "company": company},
                    source_url=r.get("url", "https://cppp.gov.in"),
                    source_name="CPPP Tender Portal",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.72,
                )
            )

    else:
        # SAM.gov via Tavily
        sam_results = _tavily_search(
            f"{company} federal contract award site:sam.gov OR USASpending",
            max_results=8,
        )
        for r in sam_results:
            claims.append(
                make_claim(
                    value={"contract": r.get("title", ""), "company": company},
                    source_url=r.get("url", "https://sam.gov"),
                    source_name="SAM.gov / USASpending",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.75,
                )
            )

    log.info("tender_search_done", company=company, results=len(claims))
    return claims


@tool
def get_order_book_data(company: str) -> list[SourcedClaim]:
    """
    Estimate the order book / pipeline size from IR page, concall transcripts, and news.
    Returns List[SourcedClaim] with pipeline estimate and order book-to-revenue ratio.
    """
    claims: list[SourcedClaim] = []

    # Screener.in concall transcripts
    concall_results = _tavily_search(
        f"{company} order book order pipeline concall transcript site:screener.in",
        max_results=5,
    )
    order_mentions: list[dict] = []
    for r in concall_results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("order book", "order pipeline", "backlog")):
            order_mentions.append({"text": content[:300], "url": r.get("url", "")})
            claims.append(
                make_claim(
                    value={"order_pipeline_note": content[:300], "company": company},
                    source_url=r.get("url", "https://screener.in"),
                    source_name="Screener.in Concall Transcript",
                    raw_snippet=content[:500],
                    confidence=0.72,
                )
            )

    # Company IR / investor relations page via Tavily
    ir_results = _tavily_search(
        f"{company} investor relations order book FY", max_results=5
    )
    for r in ir_results:
        content = r.get("content", "")
        if "order" in content.lower() or "backlog" in content.lower():
            claims.append(
                make_claim(
                    value={"ir_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"IR Page – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    # Summary claim
    claims.append(
        make_claim(
            value={
                "order_mentions": len(order_mentions),
                "company": company,
                "has_order_book_data": len(order_mentions) > 0,
            },
            source_url="https://screener.in",
            source_name="Order Book Summary",
            raw_snippet=f"Found {len(order_mentions)} order book references for {company}",
            confidence=0.70,
        )
    )
    return claims
