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
from models.sourced_claim import FinalReport

router = APIRouter()


@router.get("/report/{job_id}", response_model=ReportResponse, tags=["Analysis"])
async def get_report(job_id: str) -> ReportResponse:
    entry = job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if entry.get("status") != "done":
        raise HTTPException(status_code=202, detail=f"Job {job_id!r} is not complete yet (status: {entry.get('status')})")

    state = entry.get("state") or {}
    raw_report = state.get("final_report")
    if raw_report is None:
        raise HTTPException(status_code=500, detail="Report not available")

    # Reconstruct the Pydantic model from the JSON-deserialized dict
    try:
        final_report = FinalReport.model_validate(raw_report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report deserialization failed: {exc}")

    fr = final_report.forecast_report
    vr = final_report.verification_report

    def _proj(p) -> ForecastProjectionOut:
        # Handle different projection model field names
        if hasattr(p, "bull_crores"):
            bull, base, bear = p.bull_crores, p.base_crores, p.bear_crores
        elif hasattr(p, "bull_target"):
            bull, base, bear = p.bull_target, p.base_target, p.bear_target
        else:
            bull, base, bear = p.bull, p.base, p.bear
        return ForecastProjectionOut(
            year=p.year,
            bull=bull,
            base=base,
            bear=bear,
            confidence=getattr(p, "confidence", 0.9),
        )

    thesis_raw = fr.investment_thesis if fr else []
    thesis_str = "\n".join(thesis_raw) if isinstance(thesis_raw, list) else str(thesis_raw)

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
            time_horizon_years=fr.time_horizon_years if fr else 1,
            overall_stance=fr.overall_stance if fr else "NEUTRAL",
            confidence=fr.confidence if fr else 0.5,
            investment_thesis=thesis_str,
            key_risks=fr.key_risks if fr else [],
            key_catalysts=fr.key_catalysts if fr else [],
            eps_projections=[_proj(p) for p in (fr.eps_projections if fr else [])],
            revenue_projections=[_proj(p) for p in (fr.revenue_projections if fr else [])],
            price_projections=[_proj(p) for p in (fr.price_projections if fr else [])],
        ),
        verification=VerificationSummaryOut(
            total_claims=vr.total_claims if vr else 0,
            verified_claims=vr.verified_count if vr else 0,
            hallucinated_claims=vr.hallucinated_count if vr else 0,
            overall_confidence=vr.overall_pipeline_confidence if vr else 0.0,
        ),
        chart_data=final_report.chart_data or {},
        overall_stance=final_report.overall_stance,
        overall_confidence=final_report.overall_confidence,
    )
