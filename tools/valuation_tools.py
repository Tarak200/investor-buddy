"""
tools/valuation_tools.py
-------------------------
Fundamental valuation, technical analysis, and new-vertical EPS impact tools.

Sources: yfinance OHLCV, Screener.in, Tavily, BSE PDFs.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from langchain_core.tools import tool

from config.settings import settings
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim, ValuationTechnicalReport
from tools._base import make_claim
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)


def _safe_json_value(obj: Any) -> Any:
    """Recursively convert any dict with non-string keys (e.g. Timestamps) to be
    JSON-serializable. Call this before json.dumps when the source is unknown."""
    if isinstance(obj, dict):
        return {str(k): _safe_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json_value(i) for i in obj]
    return obj


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _yf_sym(ticker: str, market: str) -> str:
    if market.upper() == "INDIA" and not ticker.endswith((".NS", ".BO")):
        return ticker + ".NS"
    return ticker


@tool
def get_intrinsic_value_dcf(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Compute DCF intrinsic value using 5-year FCF history and WACC.
    WACC = risk-free rate (RBI/Fed 10Y) + beta × equity risk premium.
    Returns List[SourcedClaim] with intrinsic value and margin of safety %.
    """
    claims: list[SourcedClaim] = []
    sym = _yf_sym(ticker, market)

    try:
        stock = yf.Ticker(sym)
        info = stock.info or {}
        cf = stock.cashflow

        # Free Cash Flow history
        fcf_values: list[float] = []
        if cf is not None and not cf.empty:
            for col in cf.columns[:5]:
                ocf = _safe_float(cf.loc["Operating Cash Flow", col] if "Operating Cash Flow" in cf.index else 0)
                capex = _safe_float(cf.loc["Capital Expenditure", col] if "Capital Expenditure" in cf.index else 0)
                fcf_values.append(ocf + capex)  # capex is already negative

        # WACC components
        risk_free = 0.07 if market.upper() == "INDIA" else 0.045  # RBI / Fed 10Y approx
        beta = _safe_float(info.get("beta"), 1.0)
        erp = 0.055  # equity risk premium
        wacc = risk_free + beta * erp
        terminal_growth = 0.03

        # DCF calculation
        if fcf_values and any(f != 0 for f in fcf_values):
            avg_fcf = sum(fcf_values) / len(fcf_values)
            # Project 5 more years at 8% growth
            projected_fcfs = [avg_fcf * (1.08 ** i) for i in range(1, 6)]
            terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
            dcf_enterprise = sum(
                pv / (1 + wacc) ** (i + 1) for i, pv in enumerate(projected_fcfs)
            ) + terminal_value / (1 + wacc) ** 5

            shares_outstanding = _safe_float(info.get("sharesOutstanding"), 1_000_000)
            cash = _safe_float(info.get("totalCash"), 0)
            total_debt = _safe_float(info.get("totalDebt"), 0)
            dcf_equity = (dcf_enterprise + cash - total_debt)
            intrinsic_value = dcf_equity / shares_outstanding if shares_outstanding else 0.0

            current_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), 0)
            margin_of_safety = ((intrinsic_value - current_price) / intrinsic_value * 100) if intrinsic_value else 0.0

            data = {
                "dcf_intrinsic_value": round(intrinsic_value, 2),
                "dcf_margin_of_safety_pct": round(margin_of_safety, 2),
                "wacc": round(wacc * 100, 2),
                "terminal_growth": round(terminal_growth * 100, 2),
                "ticker": ticker,
            }
            snippet = json.dumps(data)[:500]
            claims.append(
                make_claim(
                    value=data,
                    source_url=f"https://finance.yahoo.com/quote/{sym}",
                    source_name="DCF Valuation (yfinance)",
                    raw_snippet=snippet,
                    confidence=0.70,
                )
            )
        else:
            log.warning("dcf_no_fcf_data", ticker=ticker)
    except Exception as exc:
        log.warning("dcf_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_relative_valuation(ticker: str, market: str, peer_list: list[str]) -> list[SourcedClaim]:
    """
    Compare company P/E, P/B, EV/EBITDA, P/S vs sector median.
    Returns List[SourcedClaim] with premium/discount %.
    """
    claims: list[SourcedClaim] = []
    sym = _yf_sym(ticker, market)

    try:
        stock = yf.Ticker(sym)
        info = stock.info or {}
        company_multiples = {
            "pe": _safe_float(info.get("trailingPE")),
            "pb": _safe_float(info.get("priceToBook")),
            "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "ps": _safe_float(info.get("priceToSalesTrailing12Months")),
        }

        peer_multiples: dict[str, list[float]] = {k: [] for k in company_multiples}
        for peer in peer_list[:8]:
            try:
                p_info = yf.Ticker(peer).info or {}
                for k, yf_key in [
                    ("pe", "trailingPE"),
                    ("pb", "priceToBook"),
                    ("ev_ebitda", "enterpriseToEbitda"),
                    ("ps", "priceToSalesTrailing12Months"),
                ]:
                    v = _safe_float(p_info.get(yf_key))
                    if v > 0:
                        peer_multiples[k].append(v)
            except Exception:
                pass

        sector_medians = {
            k: float(np.median(vals)) if vals else company_multiples[k]
            for k, vals in peer_multiples.items()
        }
        premiums = {
            k: round((company_multiples[k] - sector_medians[k]) / sector_medians[k] * 100, 2)
            if sector_medians[k] else 0.0
            for k in company_multiples
        }
        data = {
            "company_multiples": company_multiples,
            "sector_medians": sector_medians,
            "premium_discount_pct": premiums,
            "ticker": ticker,
        }
        claims.append(
            make_claim(
                value=data,
                source_url=f"https://finance.yahoo.com/quote/{sym}",
                source_name="Relative Valuation (yfinance)",
                raw_snippet=json.dumps(premiums)[:500],
                confidence=0.78,
            )
        )
    except Exception as exc:
        log.warning("relative_valuation_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_graham_number(ticker: str) -> list[SourcedClaim]:
    """
    Compute Graham Number = sqrt(22.5 × EPS × Book Value Per Share).
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        eps = _safe_float(info.get("trailingEps"))
        bvps = _safe_float(info.get("bookValue"))
        if eps > 0 and bvps > 0:
            graham = math.sqrt(22.5 * eps * bvps)
            current = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            data = {
                "graham_number": round(graham, 2),
                "current_price": round(current, 2),
                "discount_to_graham_pct": round((graham - current) / graham * 100, 2) if graham else 0.0,
                "ticker": ticker,
            }
            claims.append(
                make_claim(
                    value=data,
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    source_name="Graham Number (yfinance)",
                    raw_snippet=json.dumps(data)[:500],
                    confidence=0.80,
                )
            )
    except Exception as exc:
        log.warning("graham_number_failed", ticker=ticker, error=str(exc))
    return claims


@tool
def get_peg_ratio(ticker: str) -> list[SourcedClaim]:
    """
    Compute PEG ratio = P/E ÷ EPS growth rate.
    PEG < 1.0 flagged as potentially undervalued.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        peg = _safe_float(info.get("pegRatio") or info.get("trailingPegRatio"))
        pe = _safe_float(info.get("trailingPE"))
        eps_growth = _safe_float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))

        # Compute manually if yfinance doesn't provide it
        if peg == 0.0 and pe > 0 and eps_growth > 0:
            peg = pe / (eps_growth * 100)

        data = {
            "peg_ratio": round(peg, 3),
            "pe_ratio": round(pe, 2),
            "eps_growth_pct": round(eps_growth * 100, 2),
            "is_potentially_undervalued": 0 < peg < 1.0,
            "ticker": ticker,
        }
        claims.append(
            make_claim(
                value=data,
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                source_name="PEG Ratio (yfinance)",
                raw_snippet=json.dumps(data)[:500],
                confidence=0.75,
            )
        )
    except Exception as exc:
        log.warning("peg_ratio_failed", ticker=ticker, error=str(exc))
    return claims


@tool
def compute_valuation_verdict(claims: list[SourcedClaim]) -> list[SourcedClaim]:
    """
    Synthesise DCF, relative valuation, Graham Number, and PEG ratio into a verdict.
    Returns List[SourcedClaim] with verdict: OVERVALUED / UNDERVALUED / FAIRLY VALUED / NEUTRAL.
    """
    all_data = json.dumps(
        [_safe_json_value(c.value) for c in claims if isinstance(c.value, dict)],
        default=str,
    )[:3000]
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a quantitative equity analyst. Based on DCF, relative valuation, "
                "Graham Number, and PEG ratio data, determine the valuation verdict.\n"
                'Return JSON: {"verdict": "OVERVALUED" | "UNDERVALUED" | "FAIRLY VALUED" | "NEUTRAL", '
                '"confidence": float, "reasoning": str}'
            ),
        },
        {"role": "user", "content": all_data},
    ]
    verdict = "NEUTRAL"
    confidence = 0.5
    reasoning = ""
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=512)
        parsed = json.loads(response)
        verdict = parsed.get("verdict", "NEUTRAL")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as exc:
        log.warning("valuation_verdict_failed", error=str(exc))

    return [
        make_claim(
            value={"valuation_verdict": verdict, "confidence": confidence, "reasoning": reasoning},
            source_url="https://finance.yahoo.com",
            source_name="Valuation Verdict (LLM Synthesis)",
            raw_snippet=f"Verdict: {verdict} (confidence: {confidence:.2f})",
            confidence=confidence,
        )
    ]


@tool
def get_price_technicals(ticker: str, market: str) -> list[SourcedClaim]:
    """
    Compute technical indicators: DMA, RSI-14, MACD, Bollinger Bands, ATR, OBV,
    support/resistance from 2-year daily OHLCV via yfinance.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sym = _yf_sym(ticker, market)
    lookback = settings.technical_lookback_days

    try:
        stock = yf.Ticker(sym)
        hist = stock.history(period="2y")
        if hist is None or hist.empty or len(hist) < 50:
            log.warning("technicals_insufficient_data", ticker=ticker)
            return claims

        close = hist["Close"]
        volume = hist["Volume"]
        high = hist["High"]
        low = hist["Low"]
        current_price = float(close.iloc[-1])

        # Moving averages
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # Golden/Death cross (50 vs 200 DMA)
        golden_cross = False
        death_cross = False
        if ma200 and len(close) >= 201:
            prev_ma50 = float(close.rolling(50).mean().iloc[-2])
            prev_ma200 = float(close.rolling(200).mean().iloc[-2])
            golden_cross = prev_ma50 < prev_ma200 and ma50 > ma200
            death_cross = prev_ma50 > prev_ma200 and ma50 < ma200

        # RSI-14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.empty else 50.0

        # MACD (12/26/9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        macd_signal = "BULLISH_CROSSOVER" if macd_val > signal_val else "BEARISH_CROSSOVER"

        # Bollinger Bands (20,2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])

        # ATR-14
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # 52W high/low
        week52_high = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
        week52_low = float(close.rolling(252).min().iloc[-1]) if len(close) >= 252 else float(close.min())
        pct_from_high = round((current_price - week52_high) / week52_high * 100, 2)
        pct_from_low = round((current_price - week52_low) / week52_low * 100, 2)

        # OBV trend (simple slope)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = float(np.polyfit(range(20), obv.iloc[-20:].values, 1)[0])
        volume_trend = "ACCUMULATION" if obv_slope > 0 else "DISTRIBUTION"

        # Support/resistance from pivot points (last lookback days)
        recent = hist.tail(lookback)
        pivot = (recent["High"].mean() + recent["Low"].mean() + recent["Close"].mean()) / 3
        r1 = 2 * pivot - recent["Low"].mean()
        s1 = 2 * pivot - recent["High"].mean()
        r2 = pivot + (recent["High"].mean() - recent["Low"].mean())
        s2 = pivot - (recent["High"].mean() - recent["Low"].mean())
        r3 = recent["High"].mean() + 2 * (pivot - recent["Low"].mean())
        s3 = recent["Low"].mean() - 2 * (recent["High"].mean() - pivot)

        data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2) if ma200 else None,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "rsi_14": round(rsi, 2),
            "macd_value": round(macd_val, 4),
            "macd_signal_value": round(signal_val, 4),
            "macd_signal": macd_signal,
            "macd_bullish": macd_val > signal_val,
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "atr_14": round(atr, 2),
            "week52_high": round(week52_high, 2),
            "week52_low": round(week52_low, 2),
            "pct_from_52w_high": pct_from_high,
            "pct_from_52w_low": pct_from_low,
            "volume_trend": volume_trend,
            "obv_slope": round(obv_slope, 2),
            "resistance_levels": [round(r1, 2), round(r2, 2), round(r3, 2)],
            "support_levels": [round(s1, 2), round(s2, 2), round(s3, 2)],
            "price_vs_200dma_numeric": round((current_price - ma200) / ma200 * 100, 2) if ma200 else 0.0,
        }
        snippet = json.dumps({k: v for k, v in data.items() if k not in ("resistance_levels", "support_levels")})[:500]
        claims.append(
            make_claim(
                value=data,
                source_url=f"https://finance.yahoo.com/quote/{sym}",
                source_name="Technical Analysis (yfinance)",
                raw_snippet=snippet,
                confidence=0.88,
            )
        )
        log.info("technicals_computed", ticker=ticker, rsi=round(rsi, 2), macd=macd_signal)
    except Exception as exc:
        log.error("technicals_failed", ticker=ticker, error=str(exc))

    return claims


@tool
def get_technical_verdict(technical_claims: list[SourcedClaim]) -> list[SourcedClaim]:
    """
    Synthesise all technical signals into a verdict:
    STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL.
    Returns List[SourcedClaim].
    """
    all_data = json.dumps(
        [_safe_json_value(c.value) for c in technical_claims if isinstance(c.value, dict)],
        default=str,
    )[:3000]
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a technical analysis expert. Based on the indicators provided, "
                "produce a trading verdict.\n"
                'Return JSON: {"verdict": "STRONG BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG SELL", '
                '"confidence": float, "reasoning": str}'
            ),
        },
        {"role": "user", "content": all_data},
    ]
    verdict = "NEUTRAL"
    confidence = 0.5
    reasoning = ""
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=512)
        parsed = json.loads(response)
        verdict = parsed.get("verdict", "NEUTRAL")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as exc:
        log.warning("technical_verdict_failed", error=str(exc))

    return [
        make_claim(
            value={"technical_verdict": verdict, "confidence": confidence, "reasoning": reasoning},
            source_url="https://finance.yahoo.com",
            source_name="Technical Verdict (LLM Synthesis)",
            raw_snippet=f"Verdict: {verdict} (confidence: {confidence:.2f})",
            confidence=confidence,
        )
    ]


@tool
def get_new_verticals(company: str) -> list[SourcedClaim]:
    """
    Identify new business verticals from IR pages and concall transcripts.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    results = _tavily_search(
        f"{company} new business vertical expansion segment FY site:screener.in OR investor presentation",
        max_results=8,
    )
    for r in results:
        content = r.get("content", "")
        if any(kw in content.lower() for kw in ("new vertical", "expansion", "greenfield", "foray", "new segment")):
            claims.append(
                make_claim(
                    value={"vertical_note": content[:300], "company": company},
                    source_url=r.get("url", ""),
                    source_name=f"New Vertical – {r.get('source', 'Web')}",
                    raw_snippet=content[:500],
                    confidence=0.65,
                )
            )
    return claims


@tool
def estimate_vertical_eps_impact(company: str, verticals: list[SourcedClaim]) -> list[SourcedClaim]:
    """
    Estimate EPS impact per new vertical at Year 1 and Year 3.
    Sources labelled: Management Guidance vs Analyst Estimate vs LLM Projection.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    vertical_texts = " ".join(
        str(v.value.get("vertical_note", "")) for v in verticals if isinstance(v.value, dict)
    )[:2000]

    if not vertical_texts.strip():
        return claims

    current_year = datetime.utcnow().year
    prompt = [
        {
            "role": "system",
            "content": (
                f"Estimate EPS impact for each new business vertical mentioned. "
                f"Current year: {current_year}. "
                'Return JSON: {"verticals": [{"name": str, "description": str, '
                '"commissioning_year": int, "projected_revenue": float, '
                '"eps_impact_y1": float, "eps_impact_y3": float, '
                '"confidence": "HIGH" | "MEDIUM" | "LOW", '
                '"source_type": "Management Guidance" | "Analyst Estimate" | "LLM Projection"}],'
                '"total_eps_uplift_3y": float}'
            ),
        },
        {"role": "user", "content": f"Company: {company}\n{vertical_texts}"},
    ]
    try:
        response = get_llm_json_response(prompt, temperature=0.0, max_tokens=1024)
        parsed = json.loads(response)
        claims.append(
            make_claim(
                value={**parsed, "company": company},
                source_url="https://screener.in",
                source_name="New Vertical EPS Impact (LLM)",
                raw_snippet=json.dumps(parsed)[:500],
                confidence=0.60,
            )
        )
    except Exception as exc:
        log.warning("vertical_eps_impact_failed", error=str(exc))

    return claims
