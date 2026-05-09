"""
api/main.py
------------
FastAPI application entry point.

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze, discover, explain, health, report, status

app = FastAPI(
    title="Financial Research Copilot API",
    version="1.0.0",
    description=(
        "Multi-agent financial research assistant for US and Indian equities. "
        "Provides deep-dive analysis with 12 specialist agents, "
        "LLM-based hallucination verification, and Bull/Base/Bear projections."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_V1 = "/api/v1"
app.include_router(health.router,    prefix=_V1)
app.include_router(analyze.router,   prefix=_V1)
app.include_router(status.router,    prefix=_V1)
app.include_router(report.router,    prefix=_V1)
app.include_router(explain.router,   prefix=_V1)
app.include_router(discover.router,  prefix=_V1)
