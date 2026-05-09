"""
evaluation/eval_harness.py
----------------------------
CLI evaluation harness.  Runs the full pipeline against a golden dataset
and reports hallucination rate, latency p50/p95, and relevance score.

Usage:
  python -m evaluation.eval_harness --golden tests/eval/golden_dataset.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import structlog

from agents.orchestrator import run_analysis
from evaluation.hallucination_checker import check_claims_batch
from evaluation.latency_monitor import get_summary, reset

log = structlog.get_logger(__name__)


def _load_golden(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _relevance_score(report_md: str, expected_keywords: list[str]) -> float:
    """Simple keyword-hit relevance score (0.0–1.0)."""
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in report_md.lower())
    return hits / len(expected_keywords)


def run_eval(golden_path: str) -> dict[str, Any]:
    cases = _load_golden(golden_path)
    results: list[dict[str, Any]] = []
    reset()

    for case in cases:
        company = case["company"]
        ticker = case["ticker"]
        market = case.get("market", "INDIA")
        sector = case.get("sector", "General")
        horizon = case.get("time_horizon_years", 1)
        expected_keywords: list[str] = case.get("expected_keywords", [])

        log.info("eval_case_start", company=company, ticker=ticker)
        t0 = time.perf_counter()
        try:
            state = run_analysis(
                company=company,
                ticker=ticker,
                market=market,
                sector=sector,
                time_horizon_years=horizon,
            )
        except Exception as exc:
            log.error("eval_case_failed", company=company, error=str(exc))
            results.append({"company": company, "status": "FAILED", "error": str(exc)})
            continue

        elapsed = time.perf_counter() - t0
        final_report = state.get("final_report")
        report_md = final_report.markdown_report if final_report else ""

        # Collect all verified claims
        all_claims = [
            c
            for key in ("financial_claims", "news_claims", "legal_claims", "ownership_claims",
                        "valuation_claims", "ratings_claims")
            for c in (state.get(key) or [])
        ]

        hallucination_results = check_claims_batch(all_claims)
        hallucinated_count = sum(1 for r in hallucination_results if r.verdict == "hallucinated")
        hallucination_rate = hallucinated_count / len(hallucination_results) if hallucination_results else 0.0

        relevance = _relevance_score(report_md, expected_keywords)

        case_result = {
            "company": company,
            "ticker": ticker,
            "status": "OK",
            "elapsed_s": round(elapsed, 2),
            "total_claims": len(all_claims),
            "hallucinated_claims": hallucinated_count,
            "hallucination_rate": round(hallucination_rate, 4),
            "relevance_score": round(relevance, 4),
            "errors": len(state.get("errors", [])),
        }
        results.append(case_result)
        log.info("eval_case_done", **case_result)

    latency_summary = get_summary()
    agg = {
        "total_cases": len(cases),
        "passed": sum(1 for r in results if r.get("status") == "OK"),
        "failed": sum(1 for r in results if r.get("status") == "FAILED"),
        "avg_hallucination_rate": round(
            sum(r.get("hallucination_rate", 0) for r in results if "hallucination_rate" in r)
            / max(len([r for r in results if "hallucination_rate" in r]), 1),
            4,
        ),
        "avg_relevance_score": round(
            sum(r.get("relevance_score", 0) for r in results if "relevance_score" in r)
            / max(len([r for r in results if "relevance_score" in r]), 1),
            4,
        ),
        "latency_p50_s": latency_summary.get("financial", {}).get("p50_s", 0),
        "latency_p95_s": latency_summary.get("financial", {}).get("p95_s", 0),
        "per_case": results,
        "per_agent_latency": latency_summary,
    }
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Assistant Eval Harness")
    parser.add_argument("--golden", required=True, help="Path to golden_dataset.json")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    results = run_eval(args.golden)
    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
