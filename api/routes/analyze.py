"""
api/routes/analyze.py
----------------------
POST /api/v1/analyze
Enqueues a new analysis job and returns the job_id.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from agents.orchestrator import run_analysis
from api.schemas import AnalysisJobResponse, AnalysisRequest
from api.job_store import job_store

_REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
_REPORTS_DIR.mkdir(exist_ok=True)


def _md_to_plain(md: str) -> str:
    """Convert markdown to clean, readable plain text."""
    lines = md.splitlines()
    out = []
    for line in lines:
        # Headings: ## Foo → FOO / # Foo → FOO
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                out.append(title.upper())
                out.append("=" * len(title))
            elif level == 2:
                out.append(title.upper())
                out.append("-" * len(title))
            else:
                out.append(f"  {'  ' * (level - 3)}{title}")
            continue

        # Horizontal rules
        if re.match(r"^[-*_]{3,}\s*$", line):
            out.append("-" * 60)
            continue

        # Table rows: keep the content, strip pipes and extra spacing
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                # Separator rows like |---|---|
                if all(re.match(r"^[-:]+$", c) for c in cells):
                    out.append("")
                    continue
                out.append("  " + " | ".join(cells))
                continue

        # Remove inline markup: **bold**, *italic*, `code`, ~~strike~~
        plain_line = line
        plain_line = re.sub(r"\*\*(.+?)\*\*", r"\1", plain_line)
        plain_line = re.sub(r"\*(.+?)\*", r"\1", plain_line)
        plain_line = re.sub(r"`(.+?)`", r"\1", plain_line)
        plain_line = re.sub(r"~~(.+?)~~", r"\1", plain_line)
        # Remove markdown links: [text](url) → text
        plain_line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", plain_line)
        # Remove bare URLs (optional)
        # List bullets: - item / * item / + item → • item
        plain_line = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", plain_line)
        # Numbered lists: keep as-is

        out.append(plain_line)

    return "\n".join(out)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisJobResponse, tags=["Analysis"])
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisJobResponse:
    job_id = str(uuid.uuid4())
    job_store.create(job_id, request)

    background_tasks.add_task(
        _run_job,
        job_id=job_id,
        company=request.company,
        ticker=request.ticker,
        market=request.market,
        sector=request.sector,
        time_horizon_years=request.time_horizon_years,
    )

    return AnalysisJobResponse(job_id=job_id, status="queued", message="Analysis job queued")


def _run_job(
    job_id: str,
    company: str,
    ticker: str,
    market: str,
    sector: str,
    time_horizon_years: int,
) -> None:
    job_store.set_running(job_id)
    try:
        state = run_analysis(
            company=company,
            ticker=ticker,
            market=market,
            sector=sector,
            time_horizon_years=time_horizon_years,
            job_id=job_id,
        )
        job_store.set_done(job_id, state)
        # Save report to reports/<TICKER>.txt as plain readable text
        try:
            report_text = getattr(state, "markdown_report", None) or (
                state.get("markdown_report") if isinstance(state, dict) else None
            )
            if report_text:
                # Fix mojibake: UTF-8 bytes mis-decoded as Latin-1 (e.g. â¹ -> ₹)
                try:
                    report_text = report_text.encode("latin-1").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
                plain = _md_to_plain(report_text)
                out_path = _REPORTS_DIR / f"{ticker}.txt"
                out_path.write_text(plain, encoding="utf-8")
        except Exception:
            pass
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))
