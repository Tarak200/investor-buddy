"""
agents/specialist/news_agent.py
--------------------------------
NewsSentimentAgent — fetches recent news and enriches with LLM sentiment.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.news_tools import analyze_sentiment, get_recent_news

log = structlog.get_logger(__name__)


def run(company: str, market: str) -> tuple[list[SourcedClaim], float]:
    t0 = time.perf_counter()
    _query_hint = f"{company} recent news sentiment"
    cached = get_cached_claims(key=company, domain="NewsArticles", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []

    try:
        raw = get_recent_news.invoke({"company": company, "market": market})
        if isinstance(raw, list):
            claims.extend(raw)
    except Exception as exc:
        log.warning("news_fetch_failed", error=str(exc))

    try:
        enriched = analyze_sentiment.invoke({"articles": claims})
        if isinstance(enriched, list):
            claims = enriched
    except Exception as exc:
        log.warning("news_sentiment_failed", error=str(exc))

    elapsed = time.perf_counter() - t0
    log.info("news_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=company, domain="NewsArticles", claims=claims, query_hint=_query_hint)
    return claims, elapsed
