"""
agents/orchestrator.py
-----------------------
LangGraph StateGraph orchestrator with 7 pipeline nodes:

  validate_input → gather_parallel → verify → forecast_synthesis
    → write_report → generate_charts → finalize

12 specialist agents run concurrently in gather_parallel.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from agents.state import AnalysisState
from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)

# ── Node implementations ───────────────────────────────────────────────────────


def _validate_input(state: AnalysisState) -> AnalysisState:
    """Validate inputs; raise ValueError on bad data."""
    company = (state.get("company") or "").strip()
    ticker = (state.get("ticker") or "").strip()
    market = (state.get("market") or "").strip().upper()
    time_horizon = state.get("time_horizon_years", 1)
    sector = (state.get("sector") or "").strip()

    if not company:
        raise ValueError("company is required")
    if not ticker:
        raise ValueError("ticker is required")
    if market not in ("US", "INDIA"):
        raise ValueError(f"market must be US or INDIA, got: {market!r}")
    if time_horizon not in (1, 2, 3, 5, 7, 10):
        raise ValueError(f"time_horizon_years must be one of 1|2|3|5|7|10, got: {time_horizon}")

    return {
        **state,
        "company": company,
        "ticker": ticker.upper(),
        "market": market,
        "sector": sector or "General",
        "time_horizon_years": time_horizon,
        "job_id": state.get("job_id") or str(uuid.uuid4()),
        "errors": [],
        "agent_latencies": {},
        "current_step": "validate_input",
    }


def _gather_parallel(state: AnalysisState) -> AnalysisState:
    """Run 12 specialist agents in parallel using ThreadPoolExecutor."""
    company = state["company"]
    ticker = state["ticker"]
    market = state["market"]
    sector = state["sector"]
    latencies: dict[str, float] = dict(state.get("agent_latencies") or {})
    errors: list[str] = list(state.get("errors") or [])

    # Lazy imports to avoid circular dependencies
    from agents.specialist import (
        culture_agent,
        financial_agent,
        innovation_agent,
        legal_agent,
        management_agent,
        news_agent,
        order_book_agent,
        ownership_agent,
        peer_agent,
        product_agent,
        ratings_agent,
        valuation_agent,
    )

    task_map = {
        "financial": lambda: financial_agent.run(company, ticker, market),
        "news": lambda: news_agent.run(company, market),
        "legal": lambda: legal_agent.run(company),
        "order_book": lambda: order_book_agent.run(company, market),
        "product": lambda: product_agent.run(company, sector),
        "management": lambda: management_agent.run(company),
        "ownership": lambda: ownership_agent.run(company, ticker, market),
        "peer": lambda: peer_agent.run(company, ticker, market, sector),
        "culture": lambda: culture_agent.run(company),
        "innovation": lambda: innovation_agent.run(company, ticker, market),
        "valuation": lambda: valuation_agent.run(company, ticker, market),
        "ratings": lambda: ratings_agent.run(company, ticker, market, sector),
    }

    results: dict[str, list[SourcedClaim]] = {}

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fn): name for name, fn in task_map.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                claims, elapsed = future.result()
                results[name] = claims if isinstance(claims, list) else []
                latencies[name] = elapsed
                log.info("specialist_done", agent=name, claims=len(results[name]), elapsed=round(elapsed, 2))
            except Exception as exc:
                msg = f"{name}_agent_failed: {exc}"
                log.error(msg)
                errors.append(msg)
                results[name] = []
                latencies[name] = 0.0

    return {
        **state,
        "financial_claims": results.get("financial", []),
        "news_claims": results.get("news", []),
        "legal_claims": results.get("legal", []),
        "order_book_claims": results.get("order_book", []),
        "product_claims": results.get("product", []),
        "management_claims": results.get("management", []),
        "ownership_claims": results.get("ownership", []),
        "peer_claims": results.get("peer", []),
        "culture_claims": results.get("culture", []),
        "innovation_claims": results.get("innovation", []),
        "valuation_claims": results.get("valuation", []),
        "ratings_claims": results.get("ratings", []),
        "agent_latencies": latencies,
        "errors": errors,
        "current_step": "gather_parallel",
    }


def _verify(state: AnalysisState) -> AnalysisState:
    """Run VerificationAgent over all gathered claims."""
    from agents.verification_agent import run as verify_run

    all_claims = {
        "financial": state.get("financial_claims", []),
        "news": state.get("news_claims", []),
        "legal": state.get("legal_claims", []),
        "order_book": state.get("order_book_claims", []),
        "product": state.get("product_claims", []),
        "management": state.get("management_claims", []),
        "ownership": state.get("ownership_claims", []),
        "peer": state.get("peer_claims", []),
        "culture": state.get("culture_claims", []),
        "innovation": state.get("innovation_claims", []),
        "valuation": state.get("valuation_claims", []),
        "ratings": state.get("ratings_claims", []),
    }

    verified_by_agent, verification_report, elapsed = verify_run(all_claims)
    latencies = {**state.get("agent_latencies", {}), "verification": elapsed}

    return {
        **state,
        **{f"{k}_claims": v for k, v in verified_by_agent.items()},
        "verification_report": verification_report,
        "agent_latencies": latencies,
        "current_step": "verify",
    }


def _forecast_synthesis(state: AnalysisState) -> AnalysisState:
    """Run ForecastSynthesisAgent."""
    from agents.forecast_agent import run as forecast_run

    verified_claims = {
        "financial": state.get("financial_claims", []),
        "news": state.get("news_claims", []),
        "legal": state.get("legal_claims", []),
        "order_book": state.get("order_book_claims", []),
        "product": state.get("product_claims", []),
        "management": state.get("management_claims", []),
        "ownership": state.get("ownership_claims", []),
        "peer": state.get("peer_claims", []),
        "culture": state.get("culture_claims", []),
        "innovation": state.get("innovation_claims", []),
        "valuation": state.get("valuation_claims", []),
        "ratings": state.get("ratings_claims", []),
    }

    forecast_report, elapsed = forecast_run(
        company=state["company"],
        ticker=state["ticker"],
        time_horizon_years=state["time_horizon_years"],
        verified_claims=verified_claims,
    )
    latencies = {**state.get("agent_latencies", {}), "forecast": elapsed}

    return {
        **state,
        "forecast_report": forecast_report,
        "agent_latencies": latencies,
        "current_step": "forecast_synthesis",
    }


def _generate_charts(state: AnalysisState) -> AnalysisState:
    """Generate all Plotly charts from verified claims."""
    from tools.chart_tools import (
        create_culture_charts,
        create_financial_charts,
        create_forecast_charts,
        create_innovation_charts,
        create_management_radar,
        create_market_share_chart,
        create_ownership_chart,
        create_peer_comparison_chart,
        create_ratings_charts,
        create_risk_radar,
        create_sentiment_chart,
        create_valuation_charts,
    )
    from tools.forecast_tools import get_time_horizon_weights

    company = state["company"]
    forecast = state.get("forecast_report")
    weights: dict = {}
    try:
        w_claims = get_time_horizon_weights.invoke({"time_horizon_years": state["time_horizon_years"]})
        for c in w_claims:
            if isinstance(c.value, dict) and "weights" in c.value:
                weights = c.value["weights"]
                break
    except Exception:
        pass

    charts: dict[str, Any] = {}

    def _safe_chart(name: str, fn, **kwargs) -> None:
        try:
            charts[name] = fn.invoke(kwargs)
        except Exception as exc:
            log.warning("chart_failed", name=name, error=str(exc))

    _safe_chart("financial", create_financial_charts, financial_claims=state.get("financial_claims", []), company=company)
    _safe_chart("sentiment", create_sentiment_chart, news_claims=state.get("news_claims", []), company=company)
    _safe_chart("ownership", create_ownership_chart, ownership_claims=state.get("ownership_claims", []), company=company)
    _safe_chart("peers", create_peer_comparison_chart, peer_claims=state.get("peer_claims", []), company=company)
    _safe_chart("market_share", create_market_share_chart, market_share_claims=state.get("product_claims", []), company=company)
    _safe_chart("management", create_management_radar, management_claims=state.get("management_claims", []), company=company)
    _safe_chart("risk", create_risk_radar, legal_claims=state.get("legal_claims", []), ownership_claims=state.get("ownership_claims", []), company=company)
    _safe_chart("culture", create_culture_charts, culture_claims=state.get("culture_claims", []), company=company)
    _safe_chart("innovation", create_innovation_charts, innovation_claims=state.get("innovation_claims", []), company=company)
    _safe_chart("valuation", create_valuation_charts, valuation_claims=state.get("valuation_claims", []), company=company)
    _safe_chart("ratings", create_ratings_charts, ratings_claims=state.get("ratings_claims", []), company=company)

    if forecast:
        eps_claims = [c for claims in [state.get("valuation_claims", [])] for c in claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]
        _safe_chart(
            "forecast",
            create_forecast_charts,
            eps_claims=eps_claims,
            revenue_claims=[],
            price_claims=[],
            market_share_claims=[],
            time_horizon_weights=weights,
            company=company,
        )

    return {**state, "chart_data": charts, "current_step": "generate_charts"}


def _write_report(state: AnalysisState) -> AnalysisState:
    """Run ReportWriterAgent."""
    from agents.report_writer_agent import run as writer_run

    verified_claims = {
        k: state.get(f"{k}_claims", [])
        for k in ("financial", "news", "legal", "order_book", "product",
                  "management", "ownership", "peer", "culture", "innovation",
                  "valuation", "ratings")
    }

    final_report, elapsed = writer_run(
        company=state["company"],
        ticker=state["ticker"],
        market=state["market"],
        time_horizon_years=state["time_horizon_years"],
        verified_claims=verified_claims,
        verification_report=state["verification_report"],
        forecast_report=state["forecast_report"],
        chart_data=state.get("chart_data", {}),
    )
    latencies = {**state.get("agent_latencies", {}), "report_writer": elapsed}

    return {
        **state,
        "final_report": final_report,
        "agent_latencies": latencies,
        "current_step": "write_report",
    }


def _finalize(state: AnalysisState) -> AnalysisState:
    """Final cleanup and logging."""
    total_latency = sum(state.get("agent_latencies", {}).values())
    log.info(
        "pipeline_complete",
        job_id=state.get("job_id"),
        company=state.get("company"),
        total_latency_s=round(total_latency, 2),
        errors=len(state.get("errors", [])),
    )
    return {**state, "current_step": "finalize"}


# ── Graph construction ─────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("validate_input", _validate_input)
    graph.add_node("gather_parallel", _gather_parallel)
    graph.add_node("verify", _verify)
    graph.add_node("forecast_synthesis", _forecast_synthesis)
    graph.add_node("generate_charts", _generate_charts)
    graph.add_node("write_report", _write_report)
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "gather_parallel")
    graph.add_edge("gather_parallel", "verify")
    graph.add_edge("verify", "forecast_synthesis")
    graph.add_edge("forecast_synthesis", "generate_charts")
    graph.add_edge("generate_charts", "write_report")
    graph.add_edge("write_report", "finalize")
    graph.add_edge("finalize", END)

    return graph


# Module-level compiled graph
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def run_analysis(
    company: str,
    ticker: str,
    market: str,
    sector: str,
    time_horizon_years: int,
    job_id: str | None = None,
) -> AnalysisState:
    """
    Synchronous entry point.  Returns the final AnalysisState.
    """
    initial_state: AnalysisState = {
        "company": company,
        "ticker": ticker,
        "market": market,
        "sector": sector,
        "time_horizon_years": time_horizon_years,
        "job_id": job_id or str(uuid.uuid4()),
        "errors": [],
        "agent_latencies": {},
        "current_step": "",
    }
    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)
    return final_state
