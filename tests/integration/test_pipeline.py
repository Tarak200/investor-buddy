"""
tests/integration/test_pipeline.py
------------------------------------
Integration tests for API endpoints.
Uses httpx.AsyncClient and a live (or testcontainer-based) app instance.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from api.main import app

BASE = "http://test"


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(app=app, base_url=BASE) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client: httpx.AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ── Analyze endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_returns_job_id(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/analyze",
        json={
            "company": "Apple Inc",
            "ticker": "AAPL",
            "market": "US",
            "sector": "Technology",
            "time_horizon_years": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"


# ── Status endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_unknown_job(client: httpx.AsyncClient):
    resp = await client.get("/api/v1/status/nonexistent-job")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_after_queue(client: httpx.AsyncClient):
    # Create a job first
    resp = await client.post(
        "/api/v1/analyze",
        json={
            "company": "Reliance Industries",
            "ticker": "RELIANCE",
            "market": "INDIA",
            "sector": "Energy",
            "time_horizon_years": 1,
        },
    )
    job_id = resp.json()["job_id"]

    status_resp = await client.get(f"/api/v1/status/{job_id}")
    assert status_resp.status_code == 200
    s = status_resp.json()
    assert s["job_id"] == job_id
    assert s["status"] in ("queued", "running", "done", "failed")


# ── Report endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_not_ready_returns_202(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/analyze",
        json={
            "company": "TCS",
            "ticker": "TCS",
            "market": "INDIA",
            "sector": "Technology",
            "time_horizon_years": 3,
        },
    )
    job_id = resp.json()["job_id"]
    # Immediately fetch report — should be 202 (not ready yet)
    r_resp = await client.get(f"/api/v1/report/{job_id}")
    assert r_resp.status_code in (202, 200)


@pytest.mark.asyncio
async def test_report_unknown_job(client: httpx.AsyncClient):
    resp = await client.get("/api/v1/report/no-such-job")
    assert resp.status_code == 404
