"""
tools/innovation_tools.py
--------------------------
R&D spending, patent activity, global footprint, export trends,
competitive rankings, and certification tools.

Sources: Annual report PDFs (PyMuPDF), Screener.in, Indian Patent Office,
         USPTO, company website (Playwright), MCA21, Tavily.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.tools import tool

from config.settings import settings
from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


@tool
def get_rd_spending(company: str, ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch 5-year R&D spend (absolute + % of revenue) from annual reports.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} R&D research development expenditure annual report {market}",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("r&d", "research", "development", "innovation")):
            claims.append(
                make_claim(
                    value={"rd_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"R&D Spending – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    # Screener.in Notes to Accounts section
    clean = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{clean}/consolidated/"
    resp = safe_get(url)
    if resp:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        notes = soup.find("section", {"id": "notes"})
        if notes:
            text = notes.get_text(separator=" ", strip=True)
            if "r&d" in text.lower() or "research" in text.lower():
                claims.append(
                    make_claim(
                        value={"rd_notes_text": text[:300], "company": company},
                        source_url=url,
                        source_name="Screener.in Notes to Accounts",
                        raw_snippet=text[:500],
                        confidence=0.72,
                    )
                )

    return claims


@tool
def get_patent_activity(company: str) -> list[SourcedClaim]:
    """
    Fetch patent filing count trend and technology focus areas.
    Sources: Indian Patent Office, USPTO, Google Patents.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} patents filed patent applications technology site:ipindia.gov.in OR site:patents.google.com OR USPTO",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if "patent" in content.lower():
            claims.append(
                make_claim(
                    value={"patent_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Patent Activity – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.70,
                )
            )
    return claims


@tool
def get_global_presence(company: str) -> list[SourcedClaim]:
    """
    Map global office locations by type (HQ/Sales/Manufacturing/R&D).
    Sources: company website Locations page (Playwright) + annual report.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} global offices locations countries presence international",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("office", "location", "global", "international", "country")):
            claims.append(
                make_claim(
                    value={"global_presence_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Global Presence – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )
    return claims


@tool
def get_geographic_revenue_split(company: str) -> list[SourcedClaim]:
    """
    Fetch domestic vs export revenue % and top export countries.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} domestic export revenue geographic split annual report",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("export", "domestic", "geographic", "international revenue")):
            claims.append(
                make_claim(
                    value={"geo_revenue_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Geographic Revenue – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )
    return claims


@tool
def get_international_subsidiaries(company: str) -> list[SourcedClaim]:
    """
    List international subsidiaries from MCA21 and annual reports.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} international subsidiary overseas foreign subsidiary annual report",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("subsidiary", "subsidiary", "overseas", "foreign")):
            claims.append(
                make_claim(
                    value={"subsidiary_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Subsidiaries – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )
    return claims


@tool
def get_export_trends(company: str) -> list[SourcedClaim]:
    """
    Fetch 5-year export revenue trend. Flags > GLOBAL_PLAYER_EXPORT_THRESHOLD as "Global Player".
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    threshold = settings.global_player_export_threshold

    results = _tavily_search(
        f"{company} export revenue trend growth percentage FY",
        max_results=5,
    )
    is_global = False
    for r in results:
        content = r.get("content", "")
        if "export" in content.lower():
            import re
            pct_match = re.search(r"(\d+\.?\d*)\s*%.*export", content, re.IGNORECASE)
            export_pct = float(pct_match.group(1)) if pct_match else 0.0
            is_global = export_pct > threshold
            claims.append(
                make_claim(
                    value={
                        "export_pct": export_pct,
                        "is_global_player": is_global,
                        "company": company,
                    },
                    source_url=r.get("url", ""),
                    source_name=f"Export Trend – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )
            break

    return claims


@tool
def get_competition_rankings(company: str, products: list[str]) -> list[SourcedClaim]:
    """
    Search for Gartner Magic Quadrant, Forrester Wave, IDC, CII/FICCI/ET awards.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    queries = [
        f"{company} Gartner Magic Quadrant Forrester Wave IDC MarketScape",
        f"{company} CII FICCI ET award recognition",
        f"{company} Star Export House Navratna Miniratna Ministry of Commerce",
    ]
    for query in queries:
        results = _tavily_search(query, max_results=3)
        for r in results:
            content = r.get("content", "")
            claims.append(
                make_claim(
                    value={"award_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Industry Ranking – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )
    return claims


@tool
def get_certifications(company: str) -> list[SourcedClaim]:
    """
    Fetch quality certifications: ISO, BIS, CE, FDA, Navratna/Miniratna.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} ISO BIS CE FDA certification accreditation quality",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.upper() for kw in ("ISO", "BIS", "CE ", "FDA", "NABL", "NAVRATNA")):
            claims.append(
                make_claim(
                    value={"certification_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Certifications – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )
    return claims
