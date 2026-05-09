"""
agents/specialist/financial_agent.py
--------------------------------------
FinancialDataAgent — collects balance sheet, P&L, cash flow,
quarterly results, EPS history, and key ratios.
"""

from __future__ import annotations

import time

import structlog

from models.sourced_claim import SourcedClaim
from retrieval.claim_cache import cache_claims, get_cached_claims
from tools.financial_tools import (
    get_annual_report_data,
    get_balance_sheet,
    get_cashflow_statement,
    get_eps_history,
    get_key_ratios,
    get_pl_statement,
    get_quarterly_results,
)

log = structlog.get_logger(__name__)


def run(company: str, ticker: str, market: str) -> tuple[list[SourcedClaim], float]:
    """
    Execute the FinancialDataAgent and return (claims, elapsed_seconds).
    """
    t0 = time.perf_counter()
    _query_hint = f"{company} {ticker} financial statements balance sheet"
    cached = get_cached_claims(key=ticker, domain="FinancialData", query_hint=_query_hint)
    if cached is not None:
        return cached, time.perf_counter() - t0

    claims: list[SourcedClaim] = []
    errors: list[str] = []

    steps = [
        ("balance_sheet", lambda: get_balance_sheet.invoke({"ticker": ticker, "market": market})),
        ("pl_statement", lambda: get_pl_statement.invoke({"ticker": ticker, "market": market})),
        ("cashflow", lambda: get_cashflow_statement.invoke({"ticker": ticker, "market": market})),
        ("quarterly", lambda: get_quarterly_results.invoke({"ticker": ticker})),
        ("eps_history", lambda: get_eps_history.invoke({"ticker": ticker, "market": market})),
        ("key_ratios", lambda: get_key_ratios.invoke({"ticker": ticker, "market": market})),
        ("annual_report", lambda: get_annual_report_data.invoke({"company": company})),
    ]

    for name, fn in steps:
        try:
            result = fn()
            if isinstance(result, list):
                claims.extend(result)
        except Exception as exc:
            msg = f"FinancialDataAgent.{name} failed: {exc}"
            log.warning(msg)
            errors.append(msg)

    elapsed = time.perf_counter() - t0
    log.info("financial_agent_done", claims=len(claims), elapsed=round(elapsed, 2))
    cache_claims(key=ticker, domain="FinancialData", claims=claims, query_hint=_query_hint)
    return claims, elapsed
