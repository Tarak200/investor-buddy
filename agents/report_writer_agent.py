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
                )
            )
            idx += 1
    return footnotes


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
        for c in all_clean[:80]
    ]
    context_json = json.dumps({
        "company": company,
        "ticker": ticker,
        "market": market,
        "time_horizon_years": time_horizon_years,
        "forecast": {
            "stance": forecast_report.overall_stance,
            "thesis": forecast_report.investment_thesis,
            "key_risks": forecast_report.key_risks,
            "key_catalysts": forecast_report.key_catalysts,
        },
        "claims": claim_summaries,
    }, default=str)

    prompt_template = load_prompt("report_writer")
    user_prompt = prompt_template.format(context=context_json)

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
        markdown_report = get_llm_response(messages, temperature=0.15, max_tokens=4096)
    except Exception as exc:
        log.error("report_writer_llm_failed", error=str(exc))
        markdown_report = f"# {company} — Analysis Report\n\n*Report generation failed: {exc}*"

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
        overall_confidence=verification_report.overall_confidence,
    )

    elapsed = time.perf_counter() - t0
    log.info("report_writer_done", elapsed=round(elapsed, 2), chars=len(markdown_report))
    return final_report, elapsed
