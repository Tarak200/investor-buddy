"""
tools/web_search_tools.py
--------------------------
Product analysis, industry outlook, market share, and segment revenue tools.

Sources: Tavily, company IR page, annual report PDFs (PyMuPDF),
         IBEF reports, industry association data, ET/BS articles.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.tools import tool

from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim
from tools._base import make_claim
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


@tool
def analyze_products(company: str) -> list[SourcedClaim]:
    """
    Map the product/service portfolio with descriptions and revenue contributions.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} product portfolio segments revenue annual report",
        max_results=8,
    )
    snippets = []
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("product", "segment", "business", "division")):
            snippets.append(content[:400])
            claims.append(
                make_claim(
                    value={"product_info": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Product Info – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    # LLM structuring
    if snippets:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract product/segment information from the text. "
                    'Return JSON: {"products": [{"name": str, "description": str, '
                    '"revenue_contribution_pct": float}], '
                    '"total_products": int}'
                ),
            },
            {"role": "user", "content": " ".join(snippets[:5])[:2000]},
        ]
        try:
            response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
            parsed = json.loads(response)
            claims.append(
                make_claim(
                    value={**parsed, "company": company},
                    source_url=results[0].get("url", "") if results else "",
                    source_name="LLM-Extracted Product Portfolio",
                    raw_snippet=json.dumps(parsed)[:500],
                    confidence=0.65,
                )
            )
        except Exception as exc:
            log.warning("product_llm_failed", error=str(exc))

    return claims


@tool
def get_industry_outlook(sector: str) -> list[SourcedClaim]:
    """
    Fetch industry growth outlook and macro trends for the sector.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{sector} industry outlook growth forecast 2024 2025 IBEF CAGR",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("cagr", "growth", "market size", "forecast")):
            claims.append(
                make_claim(
                    value={"industry_note": content[:300], "sector": sector},
                    source_url=r.get("url", ""),
                    source_name=f"Industry Outlook – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    # IBEF reports specifically
    ibef_results = _tavily_search(
        f"{sector} market size India site:ibef.org",
        max_results=3,
    )
    for r in ibef_results:
        claims.append(
            make_claim(
                value={"ibef_data": r.get("content", "")[:300], "sector": sector},
                source_url=r.get("url", "https://ibef.org"),
                source_name="IBEF Industry Report",
                raw_snippet=r.get("content", "")[:500],
                confidence=0.78,
            )
        )

    return claims


@tool
def get_market_share(company: str, sector: str) -> list[SourcedClaim]:
    """
    Determine market share of the company within its sector (% basis).
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} market share {sector} percent ranking",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if "market share" in content.lower() or "%" in content:
            claims.append(
                make_claim(
                    value={"market_share_note": content[:300], "company": company, "sector": sector},
                    source_url=r.get("url", ""),
                    source_name=f"Market Share Data – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )

    return claims


@tool
def get_product_segment_revenue(company: str) -> list[SourcedClaim]:
    """
    Extract segment-wise revenue split from BSE filing PDFs or Screener.in.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    # Screener.in segment data
    results = _tavily_search(
        f"{company} segment revenue split BSE annual report site:screener.in OR BSE filing",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        claims.append(
            make_claim(
                value={"segment_data": content[:300], "company": company},
                source_url=r.get("url", "https://screener.in"),
                source_name="Screener.in Segment Revenue",
                raw_snippet=content[:500],
                confidence=0.68,
            )
        )

    return claims
