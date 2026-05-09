"""
tools/financial_tools.py
------------------------
Financial statement and ratio extraction tools.

US  : yfinance + SEC EDGAR REST API
India: Screener.in → Moneycontrol → Tickertape → BSE → NSE → yfinance (.NS)

All functions return List[SourcedClaim].
"""

from __future__ import annotations

import json
import math
from typing import Any

import structlog
import yfinance as yf
from langchain_core.tools import tool

from config.settings import settings
from models.sourced_claim import SourcedClaim
from tools._base import make_claim, safe_get

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _screener_url(ticker: str) -> str:
    """Remove .NS / .BO suffix for Screener.in lookup."""
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    return f"https://www.screener.in/company/{clean}/consolidated/"


def _yf_ticker(ticker: str, market: str) -> str:
    """Ensure .NS suffix for Indian tickers on yfinance."""
    if market.upper() == "INDIA" and not ticker.endswith((".NS", ".BO")):
        return ticker + ".NS"
    return ticker


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _compute_cagr(start: float, end: float, years: int) -> float:
    """Compound annual growth rate; returns 0 on invalid inputs."""
    if years <= 0 or start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _edgar_cik(ticker: str) -> str | None:
    url = "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
    resp = safe_get(f"https://data.sec.gov/submissions/CIK{ticker}.json")
    if resp:
        try:
            return resp.json().get("cik", None)
        except Exception:
            pass
    # Fallback: search endpoint
    search = safe_get(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": ticker, "forms": "10-K"},
    )
    if search:
        try:
            hits = search.json().get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source", {}).get("period_of_report")
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


@tool
def get_balance_sheet(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch 5-year annual balance sheet for a company.
    Returns List[SourcedClaim] with assets, liabilities, equity per year.
    """
    claims: list[SourcedClaim] = []
    yf_sym = _yf_ticker(ticker, market)

    try:
        stock = yf.Ticker(yf_sym)
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            summary = bs.iloc[:, : settings.financials_lookback_years].to_dict()
            snippet = json.dumps({str(k): str(v) for k, v in list(summary.items())[:3]})[:500]
            claims.append(
                make_claim(
                    value={"balance_sheet": summary, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{yf_sym}/balance-sheet",
                    source_name="yfinance Balance Sheet",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
            log.info("balance_sheet_fetched", ticker=ticker, source="yfinance")
    except Exception as exc:
        log.warning("balance_sheet_yfinance_failed", ticker=ticker, error=str(exc))

    # India cascade: try Screener.in if yfinance data is empty
    if not claims and market.upper() == "INDIA":
        url = _screener_url(ticker)
        resp = safe_get(url)
        if resp:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            section = soup.find("section", {"id": "balance-sheet"})
            if section:
                text = section.get_text(separator=" ", strip=True)[:500]
                claims.append(
                    make_claim(
                        value={"balance_sheet_text": text, "ticker": ticker},
                        source_url=url,
                        source_name="Screener.in Balance Sheet",
                        raw_snippet=text,
                        confidence=0.80,
                    )
                )
                log.info("balance_sheet_fetched", ticker=ticker, source="screener.in")

    if not claims:
        log.warning("balance_sheet_no_data", ticker=ticker)
    return claims


@tool
def get_pl_statement(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch 5-year annual profit & loss statement.
    Returns List[SourcedClaim] with revenue, EBITDA, net profit, margins per year.
    """
    claims: list[SourcedClaim] = []
    yf_sym = _yf_ticker(ticker, market)

    try:
        stock = yf.Ticker(yf_sym)
        income = stock.income_stmt
        if income is not None and not income.empty:
            cols = income.columns[: settings.financials_lookback_years]
            data: dict[str, Any] = {}
            for col in cols:
                year_str = str(col)[:10]
                rev = _safe_float(income.loc["Total Revenue", col] if "Total Revenue" in income.index else 0)
                net = _safe_float(income.loc["Net Income", col] if "Net Income" in income.index else 0)
                ebitda = _safe_float(income.loc["EBITDA", col] if "EBITDA" in income.index else 0)
                data[year_str] = {
                    "revenue": rev,
                    "net_income": net,
                    "ebitda": ebitda,
                    "net_margin": round(net / rev * 100, 2) if rev else 0,
                }
            snippet = json.dumps(list(data.items())[:2])[:500]
            claims.append(
                make_claim(
                    value={"pl_statement": data, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{yf_sym}/financials",
                    source_name="yfinance Income Statement",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
            log.info("pl_fetched", ticker=ticker, source="yfinance")
    except Exception as exc:
        log.warning("pl_yfinance_failed", ticker=ticker, error=str(exc))

    # India cascade
    if not claims and market.upper() == "INDIA":
        url = _screener_url(ticker)
        resp = safe_get(url)
        if resp:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            section = soup.find("section", {"id": "profit-loss"})
            if section:
                text = section.get_text(separator=" ", strip=True)[:500]
                claims.append(
                    make_claim(
                        value={"pl_text": text, "ticker": ticker},
                        source_url=url,
                        source_name="Screener.in P&L",
                        raw_snippet=text,
                        confidence=0.80,
                    )
                )

    return claims


@tool
def get_cashflow_statement(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch 5-year annual cash flow statement.
    Returns List[SourcedClaim] with operating, investing, financing cash flows.
    """
    claims: list[SourcedClaim] = []
    yf_sym = _yf_ticker(ticker, market)

    try:
        stock = yf.Ticker(yf_sym)
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            data = cf.iloc[:, : settings.financials_lookback_years].to_dict()
            snippet = json.dumps({str(k): str(list(v.values())[:3]) for k, v in list(data.items())[:2]})[:500]
            claims.append(
                make_claim(
                    value={"cashflow": data, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{yf_sym}/cash-flow",
                    source_name="yfinance Cash Flow",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
    except Exception as exc:
        log.warning("cashflow_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_quarterly_results(ticker: str) -> list[SourcedClaim]:
    """
    Fetch last N quarterly P&L results (N = settings.QUARTERLY_RESULTS_COUNT).
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    try:
        stock = yf.Ticker(ticker)
        quarterly = stock.quarterly_income_stmt
        if quarterly is not None and not quarterly.empty:
            n = settings.quarterly_results_count
            data = quarterly.iloc[:, :n].to_dict()
            snippet = json.dumps({str(k): {} for k in list(data.keys())[:3]})[:500]
            claims.append(
                make_claim(
                    value={"quarterly_results": data, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{ticker}/financials",
                    source_name="yfinance Quarterly Results",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
    except Exception as exc:
        log.warning("quarterly_results_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_eps_history(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Fetch 10-year EPS history and compute 5-year CAGR.
    Returns List[SourcedClaim] with EPS per year + CAGR.
    """
    claims: list[SourcedClaim] = []
    years = settings.eps_history_years
    yf_sym = _yf_ticker(ticker, market)

    # yfinance earnings history
    try:
        stock = yf.Ticker(yf_sym)
        earnings = stock.earnings_history
        if earnings is not None and not earnings.empty:
            recent = earnings.tail(years)
            eps_list = [
                {"period": str(row.get("period", idx)), "eps": _safe_float(row.get("epsActual", 0))}
                for idx, row in recent.iterrows()
            ]
            eps_values = [e["eps"] for e in eps_list if e["eps"] != 0]
            cagr = 0.0
            if len(eps_values) >= 2:
                cagr = _compute_cagr(abs(eps_values[0]), abs(eps_values[-1]), len(eps_values) - 1)
            snippet = json.dumps({"eps": eps_list[-3:], "cagr": round(cagr * 100, 2)})[:500]
            claims.append(
                make_claim(
                    value={"eps_history": eps_list, "eps_5y_cagr": cagr, "ticker": ticker},
                    source_url=f"https://finance.yahoo.com/quote/{yf_sym}/financials",
                    source_name="yfinance EPS History",
                    raw_snippet=snippet,
                    confidence=0.85,
                )
            )
    except Exception as exc:
        log.warning("eps_history_failed", ticker=ticker, error=str(exc))

    # India cascade: Screener.in
    if not claims and market.upper() == "INDIA":
        url = _screener_url(ticker)
        resp = safe_get(url)
        if resp:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            ratios = soup.find("section", {"id": "ratios"})
            if ratios:
                text = ratios.get_text(separator=" ", strip=True)[:500]
                claims.append(
                    make_claim(
                        value={"eps_text": text, "ticker": ticker},
                        source_url=url,
                        source_name="Screener.in Ratios",
                        raw_snippet=text,
                        confidence=0.78,
                    )
                )

    return claims


@tool
def get_annual_report_data(company: str) -> list[SourcedClaim]:
    """
    Search for and parse the latest BSE annual report PDF for an Indian company.
    Returns List[SourcedClaim] with key figures from the report.
    """
    claims: list[SourcedClaim] = []
    search_url = (
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?"
        f"pageno=1&strCat=Annual Report&strPrevDate=&strScrip=&strSearch=&strType=C"
        f"&strName={company.replace(' ', '%20')}"
    )
    resp = safe_get(search_url)
    if resp:
        try:
            data = resp.json()
            if data and isinstance(data, dict):
                snippet = json.dumps(data)[:500]
                claims.append(
                    make_claim(
                        value={"annual_report_meta": data, "company": company},
                        source_url=search_url,
                        source_name="BSE Annual Report Search",
                        raw_snippet=snippet,
                        confidence=0.70,
                    )
                )
        except Exception as exc:
            log.warning("annual_report_parse_failed", company=company, error=str(exc))

    return claims


@tool
def get_key_ratios(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Compute key financial ratios: ROE, ROA, ROCE, D/E, Current Ratio, Interest Coverage.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    yf_sym = _yf_ticker(ticker, market)

    try:
        stock = yf.Ticker(yf_sym)
        info = stock.info or {}
        ratios = {
            "roe": _safe_float(info.get("returnOnEquity")),
            "roa": _safe_float(info.get("returnOnAssets")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "profit_margin": _safe_float(info.get("profitMargins")),
            "operating_margin": _safe_float(info.get("operatingMargins")),
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "ticker": ticker,
        }
        snippet = json.dumps({k: round(v, 4) for k, v in ratios.items() if k != "ticker"})[:500]
        claims.append(
            make_claim(
                value=ratios,
                source_url=f"https://finance.yahoo.com/quote/{yf_sym}",
                source_name="yfinance Key Statistics",
                raw_snippet=snippet,
                confidence=0.85,
            )
        )
    except Exception as exc:
        log.warning("key_ratios_failed", ticker=ticker, error=str(exc))

    return claims
