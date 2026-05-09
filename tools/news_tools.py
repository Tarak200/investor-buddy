"""
tools/news_tools.py
-------------------
News fetching and sentiment analysis tools.

US  : NewsAPI + Tavily Search
India: Moneycontrol / ET / BS / Livemint / NDTV via Tavily + RSS
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import feedparser
import structlog
from langchain_core.tools import tool

from config.settings import settings
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get

log = structlog.get_logger(__name__)

# RSS feeds for Indian financial news
_INDIA_RSS_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "business_standard": "https://www.business-standard.com/rss/home_page_top_stories.rss",
    "livemint": "https://www.livemint.com/rss/markets",
    "ndtv_profit": "https://feeds.feedburner.com/ndtvprofit-latest",
}


def _tavily_search(query: str, max_results: int = 10) -> list[dict]:
    """Run a Tavily search and return raw result list."""
    if not settings.tavily_api_key:
        log.warning("tavily_key_missing")
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        result = client.search(query=query, max_results=max_results, search_depth="basic")
        return result.get("results", [])
    except Exception as exc:
        log.warning("tavily_search_failed", query=query, error=str(exc))
        return []


def _newsapi_search(company: str, days: int) -> list[dict]:
    """Fetch news from NewsAPI for a company."""
    if not settings.newsapi_key:
        return []
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": company,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": settings.newsapi_key,
        "pageSize": 50,
    }
    resp = safe_get("https://newsapi.org/v2/everything", params=params)
    if resp:
        try:
            return resp.json().get("articles", [])
        except Exception:
            pass
    return []


def _rss_fetch(company: str, days: int) -> list[dict]:
    """Fetch matching articles from Indian RSS feeds."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    articles = []
    for source, feed_url in _INDIA_RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                if company.lower() in (title + summary).lower():
                    pub_date = entry.get("published_parsed")
                    if pub_date:
                        dt = datetime(*pub_date[:6])
                        if dt >= cutoff:
                            articles.append({
                                "title": title,
                                "description": summary[:300],
                                "url": entry.get("link", feed_url),
                                "publishedAt": dt.isoformat(),
                                "source": source,
                            })
        except Exception as exc:
            log.debug("rss_fetch_error", source=source, error=str(exc))
    return articles


@tool
def get_recent_news(company: str, market: str = "INDIA") -> list[SourcedClaim]:
    """
    Gather recent news articles about the company (last NEWS_LOOKBACK_DAYS days).
    Returns List[SourcedClaim] — one claim per article.
    """
    days = settings.news_lookback_days
    claims: list[SourcedClaim] = []
    seen_urls: set[str] = set()

    # NewsAPI (US primary)
    for article in _newsapi_search(company, days):
        url = article.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        claims.append(
            make_claim(
                value={
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                },
                source_url=url,
                source_name=f"NewsAPI – {article.get('source', {}).get('name', '')}",
                raw_snippet=(article.get("description") or article.get("title", ""))[:500],
                confidence=0.80,
            )
        )

    # Tavily search (both markets)
    for result in _tavily_search(f"{company} news", max_results=15):
        url = result.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        claims.append(
            make_claim(
                value={
                    "title": result.get("title", ""),
                    "description": result.get("content", "")[:300],
                    "source": "Tavily",
                },
                source_url=url,
                source_name=f"Tavily – {result.get('source', 'Web')}",
                raw_snippet=result.get("content", "")[:500],
                confidence=0.75,
            )
        )

    # Indian RSS feeds
    if market.upper() == "INDIA":
        for article in _rss_fetch(company, days):
            url = article.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            claims.append(
                make_claim(
                    value=article,
                    source_url=url,
                    source_name=article.get("source", "Indian News RSS"),
                    raw_snippet=article.get("description", "")[:500],
                    confidence=0.75,
                )
            )

    log.info("news_fetched", company=company, articles=len(claims))
    return claims


@tool
def analyze_sentiment(articles: list[SourcedClaim]) -> list[SourcedClaim]:
    """
    Score sentiment for each article claim (−1.0 to +1.0) using LLM.
    Enriches each claim's value dict with 'sentiment_score' and 'sentiment_label'.
    Returns the enriched List[SourcedClaim].
    """
    enriched: list[SourcedClaim] = []

    for claim in articles:
        title = ""
        description = ""
        if isinstance(claim.value, dict):
            title = claim.value.get("title", "")
            description = claim.value.get("description", "")
        text = f"{title}. {description}".strip()
        if not text:
            enriched.append(claim)
            continue

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment classifier. "
                    "Given a news article snippet, return a JSON object with:\n"
                    '{"sentiment_score": float between -1.0 and 1.0, '
                    '"sentiment_label": "POSITIVE" | "NEUTRAL" | "NEGATIVE", '
                    '"topic_tags": ["tag1", "tag2"]}'
                ),
            },
            {"role": "user", "content": f"Article: {text[:800]}"},
        ]

        sentiment_score = 0.0
        sentiment_label = "NEUTRAL"
        topic_tags: list[str] = []

        try:
            response = get_llm_json_response(prompt, temperature=0.0, max_tokens=128)
            parsed = json.loads(response)
            sentiment_score = float(parsed.get("sentiment_score", 0.0))
            sentiment_label = str(parsed.get("sentiment_label", "NEUTRAL"))
            topic_tags = list(parsed.get("topic_tags", []))
        except Exception as exc:
            log.debug("sentiment_analysis_failed", error=str(exc))

        enriched_value = dict(claim.value) if isinstance(claim.value, dict) else {"raw": claim.value}
        enriched_value.update({
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "topic_tags": topic_tags,
        })
        enriched.append(
            claim.model_copy(update={"value": enriched_value})
        )

    log.info("sentiment_scored", count=len(enriched))
    return enriched
