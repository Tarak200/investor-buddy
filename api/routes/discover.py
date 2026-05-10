"""
api/routes/discover.py
-----------------------
POST /api/v1/discover
Asks: India or US?  Then discovers interesting new stocks via:
  1. SuperstarAgent   — portfolio additions by influential investors + institutions
  2. GovtSchemeAgent  — policy incentives / penalisation signals
  3. DiscoveryAgent   — consolidates, deep-dives on top candidates, returns shortlist

GET  /api/v1/discover/{job_id}
Returns the status / result of a previously queued discovery job.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from agents.discovery_agent import DiscoveryResult, StockCandidate, run_discovery
from api.job_store import job_store
from api.schemas import (
    DiscoverJobResponse,
    DiscoverRequest,
    DiscoveryResultOut,
    StockCandidateOut,
)

router = APIRouter()


# ── background task ────────────────────────────────────────────────────────────

def _run_discovery_job(job_id: str, market: str, sector: str | None, market_caps: list[str] | None) -> None:
    job_store.set_running(job_id)
    try:
        result: DiscoveryResult = run_discovery(market, sector=sector, market_caps=market_caps)
        # Serialise result into a plain-dict compatible with job_store
        job_store.set_done(job_id, _result_to_dict(result, job_id, sector=sector, market_caps=market_caps))
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))


def _candidate_to_out(c: StockCandidate) -> StockCandidateOut:
    return StockCandidateOut(
        ticker=c.ticker,
        company=c.company,
        sector=c.sector,
        market=c.market,
        superstar_conviction=c.superstar_conviction,
        policy_tailwind=c.policy_tailwind,
        financial_score=c.financial_score,
        valuation_score=c.valuation_score,
        news_sentiment=c.news_sentiment,
        composite_score=c.composite_score,
        investors_backing=c.investors_backing,
        policy_catalysts=c.policy_catalysts,
        rationale=c.rationale,
    )


def _result_to_dict(result: DiscoveryResult, job_id: str, sector: str | None = None, market_caps: list[str] | None = None) -> dict:
    return {
        "job_id": job_id,
        "market": result.market,
        "sector": sector,
        "market_caps": market_caps,
        "elapsed_seconds": result.elapsed_seconds,
        "candidates_evaluated": result.candidates_evaluated,
        "top_picks": [_candidate_to_out(c).model_dump() for c in result.top_picks],
        "errors": result.errors,
    }


# ── routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/discover",
    response_model=DiscoverJobResponse,
    tags=["Discovery"],
    summary="Discover interesting new stocks (superstar + govt scheme signals)",
)
async def start_discovery(
    request: DiscoverRequest,
    background_tasks: BackgroundTasks,
) -> DiscoverJobResponse:
    """
    Queue a discovery job.  Pass `market: "US"` or `market: "INDIA"`.

    The job will:
    - Scan superstar investor portfolio additions (India / US)
    - Scan institutional new purchases
    - Analyse government scheme / policy tailwinds
    - Deep-dive on top candidates (financial + valuation + news)
    - Return a scored shortlist of the most interesting new stocks
    """
    job_id = str(uuid.uuid4())
    job_store.create(job_id, request)

    background_tasks.add_task(
        _run_discovery_job,
        job_id=job_id,
        market=request.market,
        sector=request.sector,
        market_caps=request.market_caps,
    )

    return DiscoverJobResponse(
        job_id=job_id,
        market=request.market,
        sector=request.sector,
        market_caps=request.market_caps,
        status="queued",
        message=f"Discovery job queued for market: {request.market}",
    )


@router.get(
    "/discover/{job_id}",
    response_model=DiscoveryResultOut,
    tags=["Discovery"],
    summary="Get results of a discovery job",
)
async def get_discovery_result(job_id: str) -> DiscoveryResultOut:
    """
    Poll for the result of a previously queued discovery job.
    Returns 404 if job_id not found, 202 if still running.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.get("status") in ("queued", "running"):
        raise HTTPException(status_code=202, detail=f"Job {job_id} is still {job['status']}")
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=f"Discovery job failed: {job.get('error')}")

    result_data = job.get("state", {})
    return DiscoveryResultOut(**result_data)
