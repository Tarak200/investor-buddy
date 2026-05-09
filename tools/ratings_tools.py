"""
tools/ratings_tools.py
-----------------------
External ratings, broker recommendations, credit ratings, ESG, index memberships,
mutual fund holdings, and government scheme benefit tools.
"""

from __future__ import annotations

import json
import math

import structlog
import yfinance as yf
from langchain_core.tools import tool

from models.sourced_claim import SourcedClaim
from tools._base import make_claim
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _credit_rating_to_numeric(rating: str) -> float:
    """Map rating string to numeric score (AAA=10 ... D=1)."""
    scale = {
        "AAA": 10, "AA+": 9.5, "AA": 9, "AA-": 8.5,
        "A+": 8, "A": 7.5, "A-": 7,
        "BBB+": 6.5, "BBB": 6, "BBB-": 5.5,
        "BB+": 5, "BB": 4.5, "BB-": 4,
        "B+": 3.5, "B": 3, "B-": 2.5,
        "CCC": 2, "CC": 1.5, "C": 1.2, "D": 1,
    }
    return scale.get(rating.upper().strip(), 5.0)


@tool
def get_broker_recommendations(company: str, ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch broker analyst ratings and consensus target price.
    US: yfinance recommendations.  India: Moneycontrol + ET Markets via Tavily.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sym = ticker + ".NS" if market.upper() == "INDIA" and not ticker.endswith((".NS", ".BO")) else ticker

    # yfinance recommendations
    try:
        stock = yf.Ticker(sym)
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            recent = recs.tail(20).to_dict("records")
            buy = sum(1 for r in recent if str(r.get("To Grade", "")).upper() in ("BUY", "STRONG BUY", "OUTPERFORM", "OVERWEIGHT"))
            hold = sum(1 for r in recent if str(r.get("To Grade", "")).upper() in ("HOLD", "NEUTRAL", "EQUAL WEIGHT"))
            sell = sum(1 for r in recent if str(r.get("To Grade", "")).upper() in ("SELL", "UNDERPERFORM", "UNDERWEIGHT"))
            snippet = json.dumps(recent[:3])[:500]
            claims.append(
                make_claim(
                    value={
                        "broker_ratings": recent,
                        "buy_count": buy,
                        "hold_count": hold,
                        "sell_count": sell,
                        "ticker": ticker,
                    },
                    source_url=f"https://finance.yahoo.com/quote/{sym}/analysis",
                    source_name="yfinance Broker Recommendations",
                    raw_snippet=snippet,
                    confidence=0.80,
                )
            )
    except Exception as exc:
        log.warning("broker_recs_failed", ticker=ticker, error=str(exc))

    # India: Moneycontrol / ET via Tavily
    if market.upper() == "INDIA":
        results = _tavily_search(
            f"{company} analyst rating target price Buy Sell site:moneycontrol.com OR site:economictimes.com",
            max_results=5,
        )
        for r in results:
            content = r.get("content", "")
            claims.append(
                make_claim(
                    value={"broker_note": content[:300], "company": company},
                    source_url=r.get("url", "https://moneycontrol.com"),
                    source_name=f"Broker Research – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.72,
                )
            )

    # Target price from yfinance
    try:
        info = yf.Ticker(sym).info or {}
        target = _safe_float(info.get("targetMeanPrice"))
        current = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        upside = round((target - current) / current * 100, 2) if current else 0.0
        if target:
            claims.append(
                make_claim(
                    value={
                        "consensus_target_price": round(target, 2),
                        "consensus_upside_pct": upside,
                        "ticker": ticker,
                    },
                    source_url=f"https://finance.yahoo.com/quote/{sym}/analysis",
                    source_name="yfinance Consensus Target",
                    raw_snippet=f"Target: {target}, Upside: {upside}%",
                    confidence=0.80,
                )
            )
    except Exception:
        pass

    return claims


@tool
def get_credit_ratings(company: str, market: str) -> list[SourcedClaim]:
    """
    Fetch credit ratings from CRISIL/ICRA/CARE (India) or Moody's/S&P/Fitch (US).
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    agency_sites = (
        "site:crisil.com OR site:icra.in OR site:careratings.com"
        if market.upper() == "INDIA"
        else "Moody's OR S&P OR Fitch credit rating"
    )
    results = _tavily_search(
        f"{company} credit rating {agency_sites}",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        import re
        rating_match = re.search(r"\b(AAA|AA[+\-]?|A[+\-]?|BBB[+\-]?|BB[+\-]?|B[+\-]?|CCC|CC|C|D)\b", content, re.IGNORECASE)
        rating = rating_match.group(0).upper() if rating_match else "N/A"
        claims.append(
            make_claim(
                value={
                    "credit_rating": rating,
                    "rating_numeric": _credit_rating_to_numeric(rating),
                    "source": r.get("source", ""),
                    "company": company,
                },
                source_url=r.get("url", ""),
                source_name=f"Credit Rating – {r.get('source', 'Web')}",
                raw_snippet=content[:500],
                confidence=0.72,
            )
        )

    return claims


@tool
def get_esg_scores(company: str) -> list[SourcedClaim]:
    """
    Fetch ESG score (E, S, G components) and sector percentile rank from MSCI/Sustainalytics.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} ESG score MSCI Sustainalytics environmental social governance",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if "esg" in content.lower():
            claims.append(
                make_claim(
                    value={"esg_snippet": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"ESG Score – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )

    return claims


@tool
def get_index_memberships(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Determine which major indices the company is in and flag recent addition/removal.
    India: Nifty 50/200/500.  US: S&P 500, Russell.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    index_query = (
        f"{ticker} Nifty 50 OR Nifty 500 OR Sensex index membership"
        if market.upper() == "INDIA"
        else f"{ticker} S&P 500 OR Russell 1000 index inclusion exclusion"
    )
    results = _tavily_search(index_query, max_results=5)
    for r in results:
        content = r.get("content", "")
        claims.append(
            make_claim(
                value={"index_note": content[:300], "ticker": ticker},
                source_url=r.get("url", ""),
                source_name=f"Index Membership – {r.get('source', 'Web')}",
                raw_snippet=content[:500],
                confidence=0.70,
            )
        )
    return claims


@tool
def get_mutual_fund_holdings(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch top MF holders and QoQ net buying/selling trend.
    India: Moneycontrol MF tab + Value Research.  US: yfinance mutualfund_holders.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sym = ticker + ".NS" if market.upper() == "INDIA" and not ticker.endswith((".NS", ".BO")) else ticker

    try:
        stock = yf.Ticker(sym)
        mf = stock.mutualfund_holders
        if mf is not None and not mf.empty:
            top10 = mf.head(10).to_dict("records")
            snippet = json.dumps([{"holder": r.get("Holder", ""), "pct": str(r.get("% Out", ""))} for r in top10[:5]])[:500]
            claims.append(
                make_claim(
                    value={"mf_holders": top10, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{sym}/holders",
                    source_name="yfinance MF Holders",
                    raw_snippet=snippet,
                    confidence=0.78,
                )
            )
    except Exception as exc:
        log.warning("mf_holdings_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_government_schemes_benefit(company: str, sector: str) -> list[SourcedClaim]:
    """
    Identify PLI scheme eligibility and sector capex incentives for India.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{company} PLI scheme production linked incentive {sector} government benefit",
        max_results=5,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("pli", "scheme", "incentive", "subsidy")):
            claims.append(
                make_claim(
                    value={"scheme_note": content[:300], "company": company, "sector": sector},
                    source_url=r.get("url", ""),
                    source_name=f"Govt Scheme – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.68,
                )
            )
    return claims
