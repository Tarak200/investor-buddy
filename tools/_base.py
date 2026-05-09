"""
tools/_base.py
--------------
Shared helpers used across all tool modules:
  - make_claim()      : construct a SourcedClaim safely
  - rotating headers  : User-Agent rotation for scraping
  - safe_get()        : requests.get with retry + timeout
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def random_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,
)
def safe_get(url: str, *, timeout: int = 20, params: dict | None = None) -> requests.Response | None:
    """HTTP GET with rotating User-Agent and retry. Returns None on final failure."""
    try:
        resp = requests.get(url, headers=random_headers(), timeout=timeout, params=params)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        log.warning("http_get_failed", url=url, error=str(exc))
        return None


def make_claim(
    value: Any,
    source_url: str,
    source_name: str,
    raw_snippet: str,
    confidence: float = 0.7,
) -> SourcedClaim:
    """Factory for SourcedClaim; truncates snippet automatically."""
    snippet = str(raw_snippet)[:500] if raw_snippet else ""
    return SourcedClaim(
        value=value,
        source_url=source_url,
        source_name=source_name,
        fetched_at=datetime.utcnow(),
        raw_snippet=snippet,
        confidence=min(max(confidence, 0.0), 1.0),
    )
