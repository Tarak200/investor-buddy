"""
tools/peer_tools.py
--------------------
Peer comparison, benchmarking, and price performance tools.

Sources: Screener.in peer table, Moneycontrol peer tab, Tickertape, Tavily.
"""

from __future__ import annotations

import json
import math

import structlog
import yfinance as yf
from langchain_core.tools import tool

from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@tool
def get_peer_list(company: str, sector: str) -> list[SourcedClaim]:
    """
    Auto-identify peers in the same sector from Screener.in / Tavily.
    Returns List[SourcedClaim] with a peer ticker list.
    """
    claims: list[SourcedClaim] = []

    # Screener.in auto-peer table
    clean = company.upper().replace(" ", "-")
    url = f"https://www.screener.in/company/{clean}/consolidated/"
    resp = safe_get(url)
    peer_list: list[str] = []
    if resp:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        # Screener.in renders peers in a table under "Peers" heading
        peer_section = soup.find("section", {"id": "peers"})
        if peer_section:
            rows = peer_section.find_all("tr")
            for row in rows[1:11]:  # skip header, top 10 peers
                cells = row.find_all("td")
                if cells:
                    name = cells[0].get_text(strip=True)
                    if name:
                        peer_list.append(name)

    # Tavily fallback
    if not peer_list:
        results = _tavily_search(
            f"{company} peer companies sector {sector} comparable NSE BSE",
            max_results=5,
        )
        for r in results:
            content = r.get("content", "")
            claims.append(
                make_claim(
                    value={"peer_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Peer Info – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.60,
                )
            )

    snippet = json.dumps(peer_list[:10])
    claims.append(
        make_claim(
            value={"peer_list": peer_list, "company": company, "sector": sector},
            source_url=url,
            source_name="Screener.in Peer Table",
            raw_snippet=snippet[:500],
            confidence=0.75,
        )
    )
    log.info("peer_list_done", company=company, peers=len(peer_list))
    return claims


@tool
def get_peer_financials(peer_list: list[str]) -> list[SourcedClaim]:
    """
    Fetch key financials for each peer: Revenue, Net Margin, ROE, P/E, EV/EBITDA, D/E, EPS.
    Returns List[SourcedClaim] — one per peer.
    """
    claims: list[SourcedClaim] = []

    for ticker in peer_list[:10]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            peer_data = {
                "ticker": ticker,
                "pe_ratio": _safe_float(info.get("trailingPE")),
                "pb_ratio": _safe_float(info.get("priceToBook")),
                "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
                "roe": _safe_float(info.get("returnOnEquity")),
                "net_margin": _safe_float(info.get("profitMargins")),
                "revenue": _safe_float(info.get("totalRevenue")),
                "eps": _safe_float(info.get("trailingEps")),
                "debt_equity": _safe_float(info.get("debtToEquity")),
            }
            snippet = json.dumps({k: round(v, 4) for k, v in peer_data.items() if k != "ticker"})[:500]
            claims.append(
                make_claim(
                    value=peer_data,
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    source_name=f"yfinance – {ticker}",
                    raw_snippet=snippet,
                    confidence=0.82,
                )
            )
        except Exception as exc:
            log.warning("peer_financials_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_price_comparison(company_ticker: str, peer_list: list[str]) -> list[SourcedClaim]:
    """
    Compare 52-week high/low and 1-year return % for company vs each peer.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    all_tickers = [company_ticker] + list(peer_list[:9])

    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            hist = stock.history(period="1y")
            one_year_return = 0.0
            if hist is not None and not hist.empty and len(hist) >= 2:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                if start_price > 0:
                    one_year_return = (end_price - start_price) / start_price * 100

            data = {
                "ticker": ticker,
                "week52_high": _safe_float(info.get("fiftyTwoWeekHigh")),
                "week52_low": _safe_float(info.get("fiftyTwoWeekLow")),
                "current_price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                "one_year_return_pct": round(one_year_return, 2),
                "is_primary": ticker == company_ticker,
            }
            snippet = json.dumps(data)[:500]
            claims.append(
                make_claim(
                    value=data,
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    source_name=f"yfinance Price – {ticker}",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
        except Exception as exc:
            log.warning("price_comparison_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_product_price_benchmarking(company: str, products: list[str]) -> list[SourcedClaim]:
    """
    Estimate product price premium/discount vs peers using Tavily search.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []

    for product in products[:5]:
        results = _tavily_search(
            f"{company} {product} price vs competitors premium discount comparison",
            max_results=3,
        )
        for r in results:
            content = r.get("content", "")
            claims.append(
                make_claim(
                    value={"product": product, "pricing_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Product Pricing – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.60,
                )
            )

    return claims


@tool
def get_peer_market_share_comparison(company: str, sector: str) -> list[SourcedClaim]:
    """
    Compare market share of company vs each peer in the sector.
    Returns List[SourcedClaim] with market share table.
    """
    claims: list[SourcedClaim] = []

    results = _tavily_search(
        f"{sector} market share comparison ranking companies percentage",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if "market share" in content.lower() or "%" in content:
            claims.append(
                make_claim(
                    value={"market_share_comparison": content[:300], "sector": sector, "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"Market Share Comparison – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )

    return claims
