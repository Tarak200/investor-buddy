"""
api/routes/explain.py
----------------------
POST /api/v1/explain/{job_id}/{agent_name}
Triggers on-demand LIME explanation for the given agent.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents.explainability_agent import run as lime_run
from api.job_store import job_store
from api.schemas import ExplainResponse, FeatureImportanceOut
from models.sourced_claim import SourcedClaim

router = APIRouter()

_VALID_AGENTS = frozenset({
    "financial", "news", "legal", "order_book", "product",
    "management", "ownership", "peer", "culture", "innovation",
    "valuation", "ratings",
})


@router.post("/explain/{job_id}/{agent_name}", response_model=ExplainResponse, tags=["Explainability"])
async def explain(job_id: str, agent_name: str) -> ExplainResponse:
    if agent_name not in _VALID_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name!r}")

    entry = job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if entry.get("status") != "done":
        raise HTTPException(status_code=202, detail="Job not complete yet")

    state = entry.get("state") or {}
    raw_claims = state.get(f"{agent_name}_claims") or []
    claims = [SourcedClaim(**c) if isinstance(c, dict) else c for c in raw_claims]
    company = state.get("company", "")
    ticker = state.get("ticker", "")

    explanation = lime_run(agent_name=agent_name, claims=claims, company=company, ticker=ticker)

    return ExplainResponse(
        job_id=job_id,
        agent_name=agent_name,
        company=company,
        feature_importances=[
            FeatureImportanceOut(
                feature_name=fi.feature_name,
                importance=fi.importance,
                description=fi.description,
            )
            for fi in explanation.feature_importances
        ],
        local_prediction=explanation.local_prediction,
        score=explanation.score,
    )
