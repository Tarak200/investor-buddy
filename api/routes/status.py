"""
api/routes/status.py
---------------------
GET /api/v1/status/{job_id}
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.job_store import job_store
from api.schemas import AgentStatus, JobStatusResponse

router = APIRouter()

_AGENT_KEYS = (
    "financial", "news", "legal", "order_book", "product",
    "management", "ownership", "peer", "culture", "innovation",
    "valuation", "ratings",
)


@router.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Analysis"])
async def get_status(job_id: str) -> JobStatusResponse:
    entry = job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    state = entry.get("state") or {}
    latencies: dict = state.get("agent_latencies") or {}

    agent_statuses = [
        AgentStatus(
            agent_name=agent,
            status="done" if agent in latencies else ("running" if entry.get("status") == "running" else "queued"),
            elapsed_s=latencies.get(agent),
            claim_count=len(state.get(f"{agent}_claims") or []),
        )
        for agent in _AGENT_KEYS
    ]

    return JobStatusResponse(
        job_id=job_id,
        status=entry.get("status") or "queued",
        current_step=state.get("current_step") or entry.get("status") or "queued",
        completed=entry.get("status") == "done",
        errors=state.get("errors") or [],
        agent_statuses=agent_statuses,
    )
