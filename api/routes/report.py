"""
api/routes/report.py
---------------------
GET /api/v1/report/{job_id}
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.job_store import job_store
from api.schemas import (
    FootnoteOut,
    ForecastProjectionOut,
    ForecastReportOut,
    ReportResponse,
    VerificationSummaryOut,
)

router = APIRouter()


@router.get("/report/{job_id}", response_model=ReportResponse, tags=["Analysis"])
async def get_report(job_id: str) -> ReportResponse:
    entry = job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if entry.get("status") != "done":
        raise HTTPException(status_code=202, detail=f"Job {job_id!r} is not complete yet (status: {entry.get('status')})")

    state = entry.get("state") or {}
    final_report = state.get("final_report")
    if final_report is None:
        raise HTTPException(status_code=500, detail="Report not available")

    fr = final_report.forecast_report
    vr = final_report.verification_report

    def _proj(p) -> ForecastProjectionOut:
        return ForecastProjectionOut(
            year=p.year,
            bull=p.bull,
            base=p.base,
            bear=p.bear,
            confidence=p.confidence,
        )

    return ReportResponse(
        job_id=job_id,
        company=final_report.company,
        ticker=final_report.ticker,
        market=final_report.market,
        time_horizon_years=final_report.time_horizon_years,
        generated_at=final_report.generated_at,
        markdown_report=final_report.markdown_report,
        footnotes=[
            FootnoteOut(
                index=fn.index,
                source_name=fn.source_name,
                source_url=fn.source_url,
                fetched_at=fn.fetched_at,
            )
            for fn in (final_report.footnotes or [])
        ],
        forecast=ForecastReportOut(
            time_horizon_years=fr.time_horizon_years,
            overall_stance=fr.overall_stance,
            confidence=fr.confidence,
            investment_thesis=fr.investment_thesis,
            key_risks=fr.key_risks,
            key_catalysts=fr.key_catalysts,
            eps_projections=[_proj(p) for p in fr.eps_projections],
            revenue_projections=[_proj(p) for p in fr.revenue_projections],
            price_projections=[_proj(p) for p in fr.price_projections],
        ),
        verification=VerificationSummaryOut(
            total_claims=vr.total_claims,
            verified_claims=vr.verified_claims,
            hallucinated_claims=vr.hallucinated_claims,
            overall_confidence=vr.overall_confidence,
        ),
        chart_data=final_report.chart_data or {},
        overall_stance=final_report.overall_stance,
        overall_confidence=final_report.overall_confidence,
    )
