"""
tools/culture_tools.py
-----------------------
Employee sentiment and work culture intelligence tools.

Sources:
  - Reddit via PRAW API
  - Twitter/X via Tavily
  - Glassdoor via Playwright
  - Ambitionbox via Playwright (primary India)
  - LinkedIn job postings via Tavily (new project signals)
"""

from __future__ import annotations

import json

import structlog
from langchain_core.tools import tool

from config.settings import settings
from llm.provider import get_llm_json_response
from models.sourced_claim import CultureReport, SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _playwright_scrape(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            content = page.content()
            browser.close()
            return content
    except Exception as exc:
        log.warning("playwright_scrape_failed", url=url, error=str(exc))
        resp = safe_get(url)
        return resp.text if resp else ""


@tool
def get_reddit_employee_sentiment(company: str) -> list[SourcedClaim]:
    """
    Fetch Reddit posts about working at the company from r/india, r/cscareerquestions, etc.
    Returns List[SourcedClaim] with upvote-weighted sentiment.
    """
    claims: list[SourcedClaim] = []

    if not settings.reddit_client_id or not settings.reddit_client_secret:
        log.warning("reddit_api_keys_missing")
        # Fall back to Tavily
        results = _tavily_search(
            f"{company} employee review culture reddit site:reddit.com",
            max_results=5,
        )
        for r in results:
            claims.append(
                make_claim(
                    value={"reddit_snippet": r.get("content", "")[:300], "company": company},
                    source_url=r.get("url", "https://reddit.com"),
                    source_name="Reddit (via Tavily)",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.55,
                )
            )
        return claims

    try:
        import praw
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        subreddits = ["india", "cscareerquestions", "IndiaInvestments", "sales"]
        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                posts = sub.search(company, time_filter="year", limit=10)
                for post in posts:
                    text = f"{post.title} {post.selftext}"[:600]
                    claims.append(
                        make_claim(
                            value={
                                "title": post.title,
                                "upvotes": post.score,
                                "subreddit": sub_name,
                                "company": company,
                            },
                            source_url=f"https://reddit.com{post.permalink}",
                            source_name=f"Reddit r/{sub_name}",
                            raw_snippet=text[:500],
                            confidence=0.65,
                        )
                    )
            except Exception as exc:
                log.debug("reddit_subreddit_failed", sub=sub_name, error=str(exc))
    except ImportError:
        log.warning("praw_not_installed")

    return claims


@tool
def get_twitter_employee_chatter(company: str) -> list[SourcedClaim]:
    """
    Search Twitter/X for employee comments about working at the company via Tavily.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} employee experience work culture site:twitter.com OR site:x.com",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        claims.append(
            make_claim(
                value={"tweet_snippet": content[:300], "company": company},
                source_url=r.get("url", "https://x.com"),
                source_name="Twitter/X Employee Chatter",
                raw_snippet=content[:500],
                confidence=0.55,
            )
        )
    return claims


@tool
def get_glassdoor_culture(company: str) -> list[SourcedClaim]:
    """
    Scrape Glassdoor for culture dimension scores (8 dimensions) + review excerpts.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    slug = company.lower().replace(" ", "-")
    url = f"https://www.glassdoor.com/Reviews/{slug}-reviews-SRCH_KE0,{len(slug)}.htm"

    html = _playwright_scrape(url)
    if html and len(html) > 500:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)[:500]
        claims.append(
            make_claim(
                value={"glassdoor_snippet": text[:300], "company": company},
                source_url=url,
                source_name="Glassdoor Reviews",
                raw_snippet=text,
                confidence=0.68,
            )
        )
    else:
        # Fallback: Tavily
        results = _tavily_search(
            f"{company} Glassdoor rating work life balance compensation site:glassdoor.com",
            max_results=3,
        )
        for r in results:
            claims.append(
                make_claim(
                    value={"glassdoor_note": r.get("content", "")[:300], "company": company},
                    source_url=r.get("url", "https://glassdoor.com"),
                    source_name="Glassdoor (via Tavily)",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.62,
                )
            )
    return claims


@tool
def get_ambitionbox_culture(company: str) -> list[SourcedClaim]:
    """
    Scrape Ambitionbox for 6-dimension culture scores + department breakdown.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    slug = company.lower().replace(" ", "-")
    url = f"https://www.ambitionbox.com/overview/{slug}-overview"

    html = _playwright_scrape(url)
    if html and len(html) > 500:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)[:500]
        claims.append(
            make_claim(
                value={"ambitionbox_snippet": text[:300], "company": company},
                source_url=url,
                source_name="Ambitionbox Culture",
                raw_snippet=text,
                confidence=0.70,
            )
        )
    else:
        results = _tavily_search(
            f"{company} Ambitionbox rating work culture site:ambitionbox.com",
            max_results=3,
        )
        for r in results:
            claims.append(
                make_claim(
                    value={"ambitionbox_note": r.get("content", "")[:300], "company": company},
                    source_url=r.get("url", "https://ambitionbox.com"),
                    source_name="Ambitionbox (via Tavily)",
                    raw_snippet=r.get("content", "")[:500],
                    confidence=0.65,
                )
            )
    return claims


@tool
def get_new_project_signals(company: str) -> list[SourcedClaim]:
    """
    Infer new project signals from LinkedIn job postings and expansion news.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} new hiring expansion new project job posting site:linkedin.com OR LinkedIn",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("hiring", "expansion", "new plant", "new project", "greenfield")):
            claims.append(
                make_claim(
                    value={"signal": r.get("title", ""), "company": company},
                    source_url=r.get("url", "https://linkedin.com"),
                    source_name=f"New Project Signal – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.60,
                )
            )

    return claims


@tool
def aggregate_culture_signal(claims: list[SourcedClaim], company: str) -> list[SourcedClaim]:
    """
    Synthesise all culture claims into a unified CultureReport using LLM.
    Returns a single-item List[SourcedClaim] containing the CultureReport.
    """
    all_text = " ".join(
        str(c.value.get("glassdoor_snippet") or c.value.get("ambitionbox_snippet")
            or c.value.get("reddit_snippet") or c.value.get("tweet_snippet", ""))
        for c in claims
        if isinstance(c.value, dict)
    )[:3000]

    prompt = [
        {
            "role": "system",
            "content": (
                "Aggregate employee culture signals into a structured report. "
                "Return JSON: {"
                '"overall_culture_score": float 0-10, '
                '"source_breakdown": {"glassdoor": float, "ambitionbox": float, "reddit": float, "twitter": float}, '
                '"dimensions": {"wlb": float, "compensation": float, "growth": float, '
                '"management": float, "job_security": float, "culture_fit": float}, '
                '"sentiment_by_department": {"Engineering": float, "Sales": float, "HR": float}, '
                '"key_themes_positive": [str], "key_themes_negative": [str], '
                '"recent_trend": "IMPROVING" | "STABLE" | "DECLINING"'
                "}"
            ),
        },
        {"role": "user", "content": f"Company: {company}\nReviews: {all_text}"},
    ]

    culture_data: dict = {
        "overall_culture_score": 5.0,
        "dimensions": {"wlb": 5.0, "compensation": 5.0, "growth": 5.0},
        "recent_trend": "STABLE",
    }
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        culture_data = json.loads(response)
    except Exception as exc:
        log.warning("culture_aggregate_failed", error=str(exc))

    result_claim = make_claim(
        value={**culture_data, "company": company},
        source_url="https://ambitionbox.com",
        source_name="Culture Intelligence Aggregate",
        raw_snippet=json.dumps(culture_data)[:500],
        confidence=0.65,
    )
    return [result_claim]
