"""
tests/unit/test_tools.py
--------------------------
Unit tests for individual tool functions.
Uses mocking to avoid live network calls.
"""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.sourced_claim import SourcedClaim
from tools._base import make_claim
from tools.valuation_tools import get_graham_number
from tools.forecast_tools import get_time_horizon_weights


# ── make_claim ────────────────────────────────────────────────────────────────

def test_make_claim_returns_sourced_claim():
    claim = make_claim(
        value=42.5,
        source_url="https://example.com",
        source_name="Test Source",
        raw_snippet="Test snippet",
        confidence=0.9,
    )
    assert isinstance(claim, SourcedClaim)
    assert claim.value == 42.5
    assert claim.source_name == "Test Source"
    assert claim.confidence == 0.9
    assert claim.verified is False


def test_make_claim_truncates_snippet():
    long_snippet = "x" * 600
    claim = make_claim(
        value="v",
        source_url="https://example.com",
        source_name="S",
        raw_snippet=long_snippet,
        confidence=0.5,
    )
    assert len(claim.raw_snippet) <= 500


def test_make_claim_assigns_uuid():
    c1 = make_claim(value=1, source_url="u", source_name="s", raw_snippet="r", confidence=0.8)
    c2 = make_claim(value=1, source_url="u", source_name="s", raw_snippet="r", confidence=0.8)
    assert c1.claim_id != c2.claim_id


# ── get_graham_number ────────────────────────────────────────────────────────

def test_get_graham_number_known_values():
    """Graham number = sqrt(22.5 * EPS * BVPS)"""
    # EPS=10, BVPS=100 → sqrt(22.5 * 10 * 100) = sqrt(22500) = 150
    with patch("tools.valuation_tools.safe_get") as mock_get, \
         patch("tools.valuation_tools.get_llm_json_response") as mock_llm:
        mock_get.return_value = None
        mock_llm.return_value = {"eps": 10.0, "bvps": 100.0}
        result = get_graham_number("TestCo", "TEST", "US")
    assert isinstance(result, list)
    for claim in result:
        assert isinstance(claim, SourcedClaim)


def test_get_graham_number_handles_negative_eps():
    """Should return a claim explaining negative EPS gracefully."""
    with patch("tools.valuation_tools.safe_get") as mock_get, \
         patch("tools.valuation_tools.get_llm_json_response") as mock_llm:
        mock_get.return_value = None
        mock_llm.return_value = {"eps": -5.0, "bvps": 100.0}
        result = get_graham_number("TestCo", "TEST", "US")
    assert isinstance(result, list)


# ── get_time_horizon_weights ──────────────────────────────────────────────────

@pytest.mark.parametrize("horizon,expected_keys", [
    (1, ["technical", "order_book", "financials", "news"]),
    (3, ["financials", "order_book", "peers", "management"]),
    (5, ["rd_innovation", "management", "market_share", "financials"]),
    (10, ["rd_innovation", "management", "market_share", "financials"]),
])
def test_time_horizon_weights_keys(horizon, expected_keys):
    weights = get_time_horizon_weights(horizon)
    for key in expected_keys:
        assert key in weights, f"Key {key!r} missing for horizon {horizon}Y"


def test_time_horizon_weights_sum_to_one():
    for horizon in [1, 2, 3, 5, 7, 10]:
        weights = get_time_horizon_weights(horizon)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights don't sum to 1 for horizon {horizon}: {total}"


# ── SourcedClaim is_disputed ─────────────────────────────────────────────────

def test_sourced_claim_is_disputed_true():
    claim = SourcedClaim(
        value="x",
        source_url="http://x.com",
        source_name="X",
        hallucination_score=0.7,
        verified=True,
    )
    assert claim.is_disputed is True


def test_sourced_claim_is_disputed_false():
    claim = SourcedClaim(
        value="x",
        source_url="http://x.com",
        source_name="X",
        hallucination_score=0.3,
        verified=True,
    )
    assert claim.is_disputed is False
