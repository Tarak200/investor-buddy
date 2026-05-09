"""
agents/report_writer_agent.py
------------------------------
ReportWriterAgent — reads only verified=True claims with hallucination_score
below threshold, then uses the LLM to write a structured Markdown report
with footnote citations for every sentence.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import structlog

from config.settings import settings
from llm.prompt_manager import load_prompt
from llm.provider import get_llm_response
from models.sourced_claim import (
    FinalReport,
    ForecastReport,
    Footnote,
    SourcedClaim,
    VerificationReport,
)

log = structlog.get_logger(__name__)


def _filter_clean_claims(claims: list[SourcedClaim]) -> list[SourcedClaim]:
    """Keep only verified and non-hallucinated claims."""
    threshold = settings.hallucination_threshold
    return [c for c in claims if c.verified and c.hallucination_score < threshold]


def _build_footnotes(claims: list[SourcedClaim]) -> list[Footnote]:
    seen: dict[str, int] = {}
    footnotes: list[Footnote] = []
    idx = 1
    for c in claims:
        key = c.source_url
        if key not in seen:
            seen[key] = idx
            footnotes.append(
                Footnote(
                    index=idx,
                    source_name=c.source_name,
                    source_url=c.source_url,
                    fetched_at=c.fetched_at,
                    raw_snippet=c.raw_snippet[:200],
                )
            )
            idx += 1
    return footnotes


def _extract_financial_history(claims: list[SourcedClaim]) -> list[dict]:
    """Pull annual P&L / balance sheet rows from financial claims."""
    rows = []
    for c in claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        # Look for claims that have revenue/net_income keyed by year
        if "annual" in str(v).lower() or "revenue" in v or "net_income" in v or "eps" in v:
            rows.append(v)
    return rows


def _extract_peer_rows(claims: list[SourcedClaim]) -> list[dict]:
    """Pull peer financials rows (one dict per peer ticker)."""
    rows = []
    seen = set()
    for c in claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        if "ticker" in v and ("pe_ratio" in v or "net_margin" in v or "revenue" in v or "roe" in v):
            t = v.get("ticker", "")
            if t not in seen:
                seen.add(t)
                rows.append(v)
        elif "peer_list" in v:
            # raw peer list claim — skip
            continue
    return rows


def _extract_price_comparison(claims: list[SourcedClaim]) -> list[dict]:
    """Pull 52-week / 1-year return rows per ticker."""
    rows = []
    seen = set()
    for c in claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        if "ticker" in v and ("week52_high" in v or "one_year_return_pct" in v):
            t = v.get("ticker", "")
            if t not in seen:
                seen.add(t)
                rows.append(v)
    return rows


def _build_template_report(
    company: str,
    ticker: str,
    market: str,
    time_horizon_years: int,
    forecast_report: ForecastReport,
    claims: list[SourcedClaim],
    footnote_map: dict[str, int],
    verified_claims: dict | None = None,
) -> str:
    """Build a structured Markdown report from structured data without LLM."""
    stance_emoji = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➡️"}.get(
        forecast_report.overall_stance, "➡️"
    )
    lines = [
        f"# {company} ({ticker}) — Equity Analysis Report",
        f"*Market: {market} | Horizon: {time_horizon_years}Y | "
        f"Stance: {stance_emoji} **{forecast_report.overall_stance}** | "
        f"Confidence: {forecast_report.confidence:.0%}*",
        "",
    ]

    if forecast_report.investment_thesis:
        lines += ["## Investment Thesis", ""]
        for item in forecast_report.investment_thesis:
            lines.append(f"- {item}")
        lines.append("")

    if forecast_report.dominant_factors_for_horizon:
        lines += ["## Dominant Factors", ""]
        for item in forecast_report.dominant_factors_for_horizon:
            lines.append(f"- {item}")
        lines.append("")

    if forecast_report.key_catalysts:
        lines += ["## Key Catalysts", ""]
        for item in forecast_report.key_catalysts:
            lines.append(f"- {item}")
        lines.append("")

    if forecast_report.key_risks:
        lines += ["## Key Risks", ""]
        for item in forecast_report.key_risks:
            lines.append(f"- {item}")
        lines.append("")

    # ── Historical Financials ──────────────────────────────────────────────
    fin_claims = (verified_claims or {}).get("financial", [])
    # P&L trends — look for year-keyed revenue/net_income dicts
    history_rows: list[tuple[str, float, float, float]] = []
    for c in fin_claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        year = str(v.get("year", v.get("period", "")))
        rev = v.get("revenue", v.get("total_revenue", 0)) or 0
        ni = v.get("net_income", v.get("profit", 0)) or 0
        eps = v.get("eps", v.get("EPS", 0)) or 0
        if year and (rev or ni or eps):
            try:
                history_rows.append((year, float(rev), float(ni), float(eps)))
            except (TypeError, ValueError):
                pass

    if history_rows:
        history_rows.sort(key=lambda r: r[0])
        lines += ["## Historical Financials", ""]
        lines.append("| Year | Revenue | Net Income | EPS |")
        lines.append("|------|---------|------------|-----|")
        for year, rev, ni, eps in history_rows[-6:]:
            lines.append(f"| {year} | {rev:,.0f} | {ni:,.0f} | {eps:.2f} |")
        lines.append("")

    # Key Ratios snapshot
    ratio_rows: list[str] = []
    for c in fin_claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        if any(k in v for k in ("roe", "roa", "debt_equity", "current_ratio", "pe_ratio", "profit_margin")):
            parts = []
            for label, key in [("ROE", "roe"), ("ROA", "roa"), ("D/E", "debt_equity"),
                                ("Current Ratio", "current_ratio"), ("P/E", "pe_ratio"),
                                ("Net Margin", "profit_margin")]:
                val = v.get(key)
                if val is not None:
                    parts.append(f"**{label}**: {val}")
            if parts:
                ratio_rows.append(" | ".join(parts))
    if ratio_rows:
        lines += ["## Key Ratios", ""]
        for r in ratio_rows[:3]:
            lines.append(f"- {r}")
        lines.append("")

    # ── Peer Comparison ───────────────────────────────────────────────────
    peer_claims = (verified_claims or {}).get("peer", [])
    peer_rows = _extract_peer_rows(peer_claims)
    price_rows = _extract_price_comparison(peer_claims)

    if peer_rows:
        lines += ["## Peer Comparison — Financials", ""]
        lines.append("| Ticker | Revenue | Net Margin | ROE | P/E | D/E | EPS |")
        lines.append("|--------|---------|------------|-----|-----|-----|-----|")
        for p in peer_rows[:10]:
            t = p.get("ticker", "?")
            rev = p.get("revenue", 0) or 0
            nm = p.get("net_margin", 0) or 0
            roe = p.get("roe", 0) or 0
            pe = p.get("pe_ratio", 0) or 0
            de = p.get("debt_equity", 0) or 0
            eps = p.get("eps", 0) or 0
            lines.append(f"| {t} | {rev:,.0f} | {nm:.1%} | {roe:.1%} | {pe:.1f}x | {de:.2f} | {eps:.2f} |")
        lines.append("")

    if price_rows:
        lines += ["## Peer Comparison — Price Performance (1Y)", ""]
        lines.append("| Ticker | Current Price | 52W High | 52W Low | 1Y Return |")
        lines.append("|--------|--------------|----------|---------|-----------|")
        for p in sorted(price_rows, key=lambda x: 0 if x.get("is_primary") else 1)[:10]:
            t = p.get("ticker", "?")
            cp = p.get("current_price", 0) or 0
            hi = p.get("week52_high", 0) or 0
            lo = p.get("week52_low", 0) or 0
            ret = p.get("one_year_return_pct", 0) or 0
            primary = " ★" if p.get("is_primary") else ""
            lines.append(f"| {t}{primary} | {cp:.1f} | {hi:.1f} | {lo:.1f} | {ret:+.1f}% |")
        lines.append("")

    # ── EPS & Price Projections ───────────────────────────────────────────
    if forecast_report.eps_projections:
        lines += ["## EPS Projections", ""]
        lines.append("| Year | Bear EPS | Base EPS | Bull EPS |")
        lines.append("|------|----------|----------|----------|")
        for p in forecast_report.eps_projections:
            lines.append(f"| {p.year} | {p.bear:.2f} | {p.base:.2f} | {p.bull:.2f} |")
        lines.append("")

    if forecast_report.price_projections:
        lines += ["## Price Targets", ""]
        lines.append("| Year | Bear | Base | Bull |")
        lines.append("|------|------|------|------|")
        for p in forecast_report.price_projections:
            lines.append(f"| {p.year} | {p.bear_target:.0f} | {p.base_target:.0f} | {p.bull_target:.0f} |")
        lines.append("")

    # ── Sources ───────────────────────────────────────────────────────────
    if claims:
        lines += ["## Key Data Sources", ""]
        for c in claims[:20]:
            fn = footnote_map.get(c.source_url, "?")
            val = str(c.value)[:100].replace("\n", " ")
            lines.append(f"- [{fn}] {val}")
        lines.append("")

    lines += [
        "---",
        f"*Report generated automatically. {len(claims)} verified claims.*",
    ]
    return "\n".join(lines)


def run(
    company: str,
    ticker: str,
    market: str,
    time_horizon_years: int,
    verified_claims: dict[str, list[SourcedClaim]],
    verification_report: VerificationReport,
    forecast_report: ForecastReport,
    chart_data: dict,
) -> tuple[FinalReport, float]:
    t0 = time.perf_counter()

    # Gather clean claims
    all_clean: list[SourcedClaim] = []
    for agent, claims in verified_claims.items():
        all_clean.extend(_filter_clean_claims(claims))

    footnotes = _build_footnotes(all_clean)
    footnote_map = {f.source_url: f.index for f in footnotes}

    # Prepare prompt context (cap to avoid LLM context limits)
    claim_summaries = [
        {"fn": footnote_map.get(c.source_url, "?"), "value": str(c.value)[:150]}
        for c in all_clean[:60]
    ]

    # Build structured peer + financial summaries for the LLM prompt
    peer_rows = _extract_peer_rows(verified_claims.get("peer", []))
    price_rows = _extract_price_comparison(verified_claims.get("peer", []))
    fin_claims = verified_claims.get("financial", [])
    history_rows: list[dict] = []
    for c in fin_claims:
        v = c.value
        if not isinstance(v, dict):
            continue
        year = str(v.get("year", v.get("period", "")))
        rev = v.get("revenue", v.get("total_revenue", 0)) or 0
        ni = v.get("net_income", v.get("profit", 0)) or 0
        eps = v.get("eps", v.get("EPS", 0)) or 0
        if year and (rev or ni or eps):
            try:
                history_rows.append({"year": year, "revenue": float(rev), "net_income": float(ni), "eps": float(eps)})
            except (TypeError, ValueError):
                pass

    verified_data_summary = json.dumps(claim_summaries, default=str)
    forecast_summary = json.dumps({
        "stance": forecast_report.overall_stance,
        "confidence": forecast_report.confidence,
        "thesis": forecast_report.investment_thesis,
        "key_risks": forecast_report.key_risks,
        "key_catalysts": forecast_report.key_catalysts,
        "dominant_factors": forecast_report.dominant_factors_for_horizon,
        "time_horizon_years": time_horizon_years,
        "peer_financials": peer_rows[:8],
        "peer_price_performance": price_rows[:8],
        "historical_financials": sorted(history_rows, key=lambda r: r["year"])[-6:],
    }, default=str)

    prompt_template = load_prompt("report_writer")
    user_prompt = prompt_template.format(
        ticker=ticker,
        market=market,
        verified_data_summary=verified_data_summary,
        forecast_summary=forecast_summary,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst writing an institutional-grade report. "
                "Every factual sentence must end with a footnote marker [N] where N is the index. "
                "Use the provided footnote indices."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    markdown_report = ""
    try:
        # Truncate user prompt to avoid TPM limits
        user_msg = messages[-1]["content"]
        if len(user_msg) > 4000:
            messages[-1] = {**messages[-1], "content": user_msg[:4000]}
        # Pause to let TPM quota recover after the verification batch flood
        time.sleep(45)
        markdown_report = get_llm_response(messages, temperature=0.15, max_tokens=2048)
    except Exception as exc:
        log.warning("report_writer_llm_failed_using_template", error=str(exc)[:200])
        markdown_report = _build_template_report(
            company, ticker, market, time_horizon_years, forecast_report, all_clean,
            footnote_map, verified_claims=verified_claims,
        )

    # Build FinalReport
    final_report = FinalReport(
        company=company,
        ticker=ticker,
        market=market,
        time_horizon_years=time_horizon_years,
        generated_at=datetime.utcnow(),
        markdown_report=markdown_report,
        footnotes=footnotes,
        verification_report=verification_report,
        forecast_report=forecast_report,
        chart_data=chart_data,
        overall_stance=forecast_report.overall_stance,
        overall_confidence=verification_report.overall_pipeline_confidence,
    )

    elapsed = time.perf_counter() - t0
    log.info("report_writer_done", elapsed=round(elapsed, 2), chars=len(markdown_report))
    return final_report, elapsed
