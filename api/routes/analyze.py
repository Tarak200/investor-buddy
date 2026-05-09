"""
api/routes/analyze.py
----------------------
POST /api/v1/analyze
Enqueues a new analysis job and returns the job_id.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks

from agents.orchestrator import run_analysis
from api.schemas import AnalysisJobResponse, AnalysisRequest
from api.job_store import job_store

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
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))
