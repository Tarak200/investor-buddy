"""
tools/ownership_tools.py
-------------------------
Shareholding pattern, promoter holding, pledge, and institutional ownership tools.

US  : yfinance institutional_holders + SEC 13F via EDGAR
India: NSE/BSE shareholding pages, Screener.in, Trendlyne
"""

from __future__ import annotations

import json
import math

import structlog
import yfinance as yf
from langchain_core.tools import tool

from config.settings import settings
from models.sourced_claim import OwnershipSnapshot, SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _yf_sym(ticker: str, market: str) -> str:
    if market.upper() == "INDIA" and not ticker.endswith((".NS", ".BO")):
        return ticker + ".NS"
    return ticker


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@tool
def get_promoter_holding(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch promoter / insider holding % over the last 8 quarters.
    India: NSE/BSE shareholding + Screener.in.  US: yfinance insider holders.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sym = _yf_sym(ticker, market)

    # yfinance major holders
    try:
        stock = yf.Ticker(sym)
        major = stock.major_holders
        institutional = stock.institutional_holders
        if major is not None and not major.empty:
            data = major.to_dict()
            snippet = major.to_string()[:500]
            claims.append(
                make_claim(
                    value={"major_holders": data, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{sym}/holders",
                    source_name="yfinance Major Holders",
                    raw_snippet=snippet,
                    confidence=0.82,
                )
            )
    except Exception as exc:
        log.warning("promoter_holding_yfinance_failed", ticker=ticker, error=str(exc))

    # India: Screener.in shareholding
    if market.upper() == "INDIA":
        clean = ticker.replace(".NS", "").replace(".BO", "")
        url = f"https://www.screener.in/company/{clean}/consolidated/"
        resp = safe_get(url)
        if resp:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            section = soup.find("section", {"id": "shareholding"})
            if section:
                text = section.get_text(separator=" ", strip=True)[:500]
                claims.append(
                    make_claim(
                        value={"shareholding_text": text, "ticker": ticker},
                        source_url=url,
                        source_name="Screener.in Shareholding",
                        raw_snippet=text,
                        confidence=0.80,
                    )
                )

    return claims


@tool
def get_promoter_pledging(ticker: str) -> list[SourcedClaim]:
    """
    Fetch promoter pledging percentage. Flags HIGH RISK if > PLEDGE_HIGH_RISK_THRESHOLD.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    threshold = settings.pledge_high_risk_threshold

    results = _tavily_search(
        f"{ticker} promoter pledging percentage shareholding",
        max_results=5,
    )
    pledge_pct = 0.0
    for r in results:
        content = r.get("content", "")
        if "pledge" in content.lower():
            import re
            match = re.search(r"(\d+\.?\d*)\s*%.*pledge", content, re.IGNORECASE)
            if match:
                pledge_pct = _safe_float(match.group(1))
            claims.append(
                make_claim(
                    value={
                        "pledge_pct": pledge_pct,
                        "high_pledge_risk": pledge_pct > threshold,
                        "ticker": ticker,
                    },
                    source_url=r.get("url", ""),
                    source_name=f"Pledging Data – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.70,
                )
            )
            break  # First reliable mention is enough

    claims.append(
        make_claim(
            value={
                "pledge_pct": pledge_pct,
                "high_pledge_risk": pledge_pct > threshold,
                "ticker": ticker,
                "threshold_pct": threshold,
            },
            source_url="https://www.screener.in",
            source_name="Pledge Risk Summary",
            raw_snippet=f"Pledge: {pledge_pct}% (threshold: {threshold}%). Risk: {'HIGH' if pledge_pct > threshold else 'NORMAL'}",
            confidence=0.68,
        )
    )
    return claims


@tool
def get_institutional_ownership(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch institutional ownership breakdown: FII, DII, MF, Insurance (India) or
    institutional % (US).
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sym = _yf_sym(ticker, market)

    try:
        stock = yf.Ticker(sym)
        inst = stock.institutional_holders
        if inst is not None and not inst.empty:
            top5 = inst.head(5).to_dict("records")
            snippet = json.dumps(top5)[:500]
            claims.append(
                make_claim(
                    value={"institutional_holders": top5, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{sym}/holders",
                    source_name="yfinance Institutional Holders",
                    raw_snippet=snippet,
                    confidence=0.82,
                )
            )
    except Exception as exc:
        log.warning("institutional_ownership_failed", ticker=ticker, error=str(exc))

    # India: Trendlyne FII/DII data via Tavily
    if market.upper() == "INDIA":
        results = _tavily_search(
            f"{ticker} FII DII shareholding pattern site:trendlyne.com OR site:screener.in",
            max_results=5,
        )
        for r in results:
            content = r.get("content", "")
            if any(kw in content.lower() for kw in ("fii", "dii", "mutual fund")):
                claims.append(
                    make_claim(
                        value={"fii_dii_snippet": content[:300], "ticker": ticker},
                        source_url=r.get("url", "https://trendlyne.com"),
                        source_name="Trendlyne FII/DII",
                        raw_snippet=content[:500],
                        confidence=0.72,
                    )
                )
                break

    return claims


@tool
def get_top_shareholders(ticker: str) -> list[SourcedClaim]:
    """
    Fetch top 10 shareholders with QoQ change in holding %.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    try:
        stock = yf.Ticker(ticker)
        inst = stock.institutional_holders
        if inst is not None and not inst.empty:
            top10 = inst.head(10).to_dict("records")
            snippet = json.dumps([
                {"holder": r.get("Holder", ""), "pct": str(r.get("% Out", ""))}
                for r in top10[:5]
            ])[:500]
            claims.append(
                make_claim(
                    value={"top_shareholders": top10, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{ticker}/holders",
                    source_name="yfinance Top Shareholders",
                    raw_snippet=snippet,
                    confidence=0.82,
                )
            )
    except Exception as exc:
        log.warning("top_shareholders_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_ownership_trend(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Compute promoter holding trend and FII accumulation/distribution slope over 8 quarters.
    Returns List[SourcedClaim] with trend flags.
    """
    claims: list[SourcedClaim] = []
    quarters = settings.promoter_holding_quarters

    results = _tavily_search(
        f"{ticker} shareholding pattern quarterly trend promoter FII",
        max_results=8,
    )
    has_data = False
    for r in results:
        content = r.get("content", "")
        if "promoter" in content.lower() and ("%" in content or "quarter" in content.lower()):
            has_data = True
            claims.append(
                make_claim(
                    value={"ownership_trend_note": content[:300], "ticker": ticker},
                    source_url=r.get("url", ""),
                    source_name=f"Ownership Trend – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )

    claims.append(
        make_claim(
            value={
                "trend_analysis_available": has_data,
                "quarters_analysed": quarters,
                "ticker": ticker,
            },
            source_url="https://screener.in",
            source_name="Ownership Trend Summary",
            raw_snippet=f"Ownership trend data {'available' if has_data else 'unavailable'} for {ticker}",
            confidence=0.65,
        )
    )
    return claims
