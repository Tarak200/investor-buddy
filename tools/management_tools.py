"""
tools/management_tools.py
--------------------------
Management profile, board composition, and employee review tools.

Sources: BSE annual report PDFs, company website (Playwright),
         Ambitionbox (primary India), LinkedIn via Tavily, MCA21 directors.
"""

from __future__ import annotations

import json
import re

import structlog
from langchain_core.tools import tool

from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _scrape_with_playwright(url: str) -> str:
    """Use Playwright to get page HTML (falls back to requests on failure)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000)
            content = page.content()
            browser.close()
            return content
    except Exception as exc:
        log.warning("playwright_failed", url=url, error=str(exc))
        resp = safe_get(url)
        return resp.text if resp else ""


@tool
def get_management_profiles(company: str) -> list[SourcedClaim]:
    """
    Fetch executive profiles (name, role, tenure, education, prior roles).
    Sources: BSE annual report + company website + LinkedIn via Tavily.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    # Tavily for management bio
    results = _tavily_search(
        f"{company} CEO MD management team executive leadership",
        max_results=8,
    )
    profiles_found: list[dict] = []
    for r in results:
        content = r.get("content", "")
        if any(title in content.lower() for title in ("ceo", "md", "managing director", "chairman", "president")):
            profiles_found.append({"bio": content[:300], "url": r.get("url", "")})
            claims.append(
                make_claim(
                    value={"profile_snippet": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Management Profile – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.70,
                )
            )

    # LLM extraction of structured profiles
    if profiles_found:
        combined_text = " ".join(p["bio"] for p in profiles_found[:5])
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract executive profiles from the text. "
                    'Return JSON: {"executives": [{"name": str, "role": str, '
                    '"tenure_years": float, "education": str, "prior_roles": [str]}]}'
                ),
            },
            {"role": "user", "content": combined_text[:2000]},
        ]
        try:
            response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
            parsed = json.loads(response)
            executives = parsed.get("executives", [])
            for exec_profile in executives:
                claims.append(
                    make_claim(
                        value={**exec_profile, "company": company},
                        source_url=profiles_found[0]["url"],
                        source_name="LLM-Extracted Management Profile",
                        raw_snippet=json.dumps(exec_profile)[:500],
                        confidence=0.68,
                    )
                )
        except Exception as exc:
            log.warning("management_llm_extraction_failed", error=str(exc))

    log.info("management_profiles_done", company=company, profiles=len(profiles_found))
    return claims


@tool
def get_employee_reviews(company: str) -> list[SourcedClaim]:
    """
    Fetch employee review scores from Ambitionbox (primary) and Glassdoor via Tavily.
    Returns List[SourcedClaim] with dimension scores and overall rating.
    """
    claims: list[SourcedClaim] = []

    # Ambitionbox via Playwright
    ambitionbox_url = f"https://www.ambitionbox.com/overview/{company.lower().replace(' ', '-')}-overview"
    html = _scrape_with_playwright(ambitionbox_url)
    if html and len(html) > 500:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        rating_tag = soup.find("span", class_=re.compile(r"rating", re.I))
        rating = rating_tag.get_text(strip=True) if rating_tag else "N/A"
        snippet = f"Ambitionbox overall rating for {company}: {rating}"
        claims.append(
            make_claim(
                value={"ambitionbox_rating": rating, "company": company},
                source_url=ambitionbox_url,
                source_name="Ambitionbox Employee Reviews",
                raw_snippet=snippet,
                confidence=0.72,
            )
        )

    # Glassdoor via Tavily
    glassdoor_results = _tavily_search(
        f"{company} Glassdoor employee review rating site:glassdoor.com",
        max_results=3,
    )
    for r in glassdoor_results:
        content = r.get("content", "")
        if "rating" in content.lower() or "review" in content.lower():
            claims.append(
                make_claim(
                    value={"glassdoor_snippet": content[:300], "company": company},
                    source_url=r.get("url", "https://glassdoor.com"),
                    source_name="Glassdoor Reviews",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    return claims


@tool
def get_board_qualifications(company: str) -> list[SourcedClaim]:
    """
    Profile the board of directors: independence ratio, avg tenure, qualifications.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    board_results = _tavily_search(
        f"{company} board of directors independent directors annual report",
        max_results=5,
    )
    for r in board_results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("independent director", "board", "chairman")):
            claims.append(
                make_claim(
                    value={"board_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Board Governance – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )

    # LLM extraction
    if claims:
        combined = " ".join(
            str(c.value.get("board_note", "")) for c in claims[:3]
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract board composition from the text. "
                    'Return JSON: {"board_size": int, "independent_count": int, '
                    '"avg_tenure_years": float, "independence_ratio": float, '
                    '"board_members": [{"name": str, "role": str}]}'
                ),
            },
            {"role": "user", "content": combined[:2000]},
        ]
        try:
            response = get_llm_json_response(prompt, temperature=0.0, max_tokens=512)
            parsed = json.loads(response)
            claims.append(
                make_claim(
                    value={**parsed, "company": company},
                    source_url=claims[0].source_url,
                    source_name="LLM-Extracted Board Composition",
                    raw_snippet=json.dumps(parsed)[:500],
                    confidence=0.65,
                )
            )
        except Exception as exc:
            log.warning("board_llm_extraction_failed", error=str(exc))

    return claims
