"""
api/routes/health.py
---------------------
GET /api/v1/health
"""

from datetime import datetime

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.utcnow())
