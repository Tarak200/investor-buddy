"""
evaluation/benchmark.py
------------------------
Structured benchmark runner that wraps eval_harness.run_eval and produces:

  - Per-case composite score across 6 dimensions  (0–100)
  - Quality-gate pass/fail per case
  - Aggregate summary with letter grade
  - Historical result saving  →  tests/eval/benchmark_results/
  - Delta comparison against the previous saved run
  - Formatted score card printed to stdout

Usage
-----
  # Run all cases, print score card
  python -m evaluation.benchmark

  # Run and save results to disk
  python -m evaluation.benchmark --save

  # Run, save, and compare against last saved run
  python -m evaluation.benchmark --save --compare

  # Custom golden dataset or thresholds
  python -m evaluation.benchmark --golden path/to/golden.json --thresholds path/to/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from evaluation.eval_harness import run_eval

log = structlog.get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DEFAULT_GOLDEN = _ROOT / "tests" / "eval" / "golden_dataset.json"
_DEFAULT_THRESHOLDS = _ROOT / "tests" / "eval" / "benchmark_thresholds.json"
_RESULTS_DIR = _ROOT / "tests" / "eval" / "benchmark_results"

# ── Default quality-gate thresholds ──────────────────────────────────────────
_BUILTIN_THRESHOLDS: dict[str, float] = {
    "hallucination_rate_max": 0.10,   # ≤10 % of claims flagged as hallucinated
    "relevance_score_min": 0.60,      # ≥60 % keyword hit rate
    "attribution_coverage_min": 0.70, # ≥70 % of claims have source_url + raw_snippet
    "claim_yield_min": 3,             # ≥3 total claims per case
    "source_diversity_min": 2,        # ≥2 unique source domains
    "composite_score_min": 50.0,      # ≥50 / 100 composite score
}

# ── Dimension weights (must sum to 1.0) ───────────────────────────────────────
_WEIGHTS: dict[str, float] = {
    "hallucination": 0.30,
    "relevance":     0.25,
    "attribution":   0.20,
    "yield":         0.10,
    "errors":        0.10,
    "diversity":     0.05,
}


# ── Per-dimension raw → 0–1 scorers ──────────────────────────────────────────

def _dim_hallucination(hallucination_rate: float) -> float:
    """Lower hallucination rate is better. 0 % → 1.0, 100 % → 0.0."""
    return max(0.0, 1.0 - hallucination_rate)


def _dim_relevance(relevance_score: float) -> float:
    return float(relevance_score)


def _dim_attribution(attribution_coverage: float) -> float:
    return float(attribution_coverage)


def _dim_yield(total_claims: int) -> float:
    """Scale 0 – 15 claims linearly to 0.0 – 1.0."""
    return min(1.0, total_claims / 15.0)


def _dim_errors(errors: int) -> float:
    """0 errors → 1.0, ≥5 errors → 0.0."""
    return max(0.0, 1.0 - (errors / 5.0))


def _dim_diversity(unique_sources: int) -> float:
    """Scale 0 – 5 unique domains linearly to 0.0 – 1.0."""
    return min(1.0, unique_sources / 5.0)


# ── Composite score ────────────────────────────────────────────────────────────

def _composite_score(dims: dict[str, float]) -> float:
    """Weighted sum of dimension scores scaled to 0 – 100."""
    raw = (
        dims["hallucination"] * _WEIGHTS["hallucination"]
        + dims["relevance"]   * _WEIGHTS["relevance"]
        + dims["attribution"] * _WEIGHTS["attribution"]
        + dims["yield"]       * _WEIGHTS["yield"]
        + dims["errors"]      * _WEIGHTS["errors"]
        + dims["diversity"]   * _WEIGHTS["diversity"]
    )
    return round(raw * 100.0, 1)


def _letter_grade(composite: float) -> str:
    if composite >= 90:
        return "A+"
    if composite >= 80:
        return "A"
    if composite >= 70:
        return "B"
    if composite >= 60:
        return "C"
    if composite >= 50:
        return "D"
    return "F"


# ── Quality gates ─────────────────────────────────────────────────────────────

def _check_gates(case_score: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Return a list of quality-gate failure descriptions for one case."""
    failures: list[str] = []
    raw = case_score["raw"]
    dims = case_score["dims"]

    if raw["hallucination_rate"] > thresholds["hallucination_rate_max"]:
        failures.append(
            f"hallucination_rate {raw['hallucination_rate']:.1%} > max {thresholds['hallucination_rate_max']:.1%}"
        )
    if raw["relevance_score"] < thresholds["relevance_score_min"]:
        failures.append(
            f"relevance {raw['relevance_score']:.1%} < min {thresholds['relevance_score_min']:.1%}"
        )
    if dims["attribution"] < thresholds["attribution_coverage_min"]:
        failures.append(
            f"attribution {dims['attribution']:.1%} < min {thresholds['attribution_coverage_min']:.1%}"
        )
    if raw["total_claims"] < thresholds["claim_yield_min"]:
        failures.append(
            f"claim_yield {raw['total_claims']} < min {int(thresholds['claim_yield_min'])}"
        )
    if raw["unique_sources"] < thresholds["source_diversity_min"]:
        failures.append(
            f"source_diversity {raw['unique_sources']} < min {int(thresholds['source_diversity_min'])}"
        )
    if case_score["composite"] < thresholds["composite_score_min"]:
        failures.append(
            f"composite {case_score['composite']:.1f} < min {thresholds['composite_score_min']:.1f}"
        )
    return failures


# ── Scoring a full eval result ────────────────────────────────────────────────

def score_eval_result(eval_result: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """
    Convert the raw dict returned by eval_harness.run_eval into a fully
    scored benchmark payload with per-case composites and an aggregate summary.
    """
    scored_cases: list[dict[str, Any]] = []

    for case in eval_result.get("per_case", []):
        if case.get("status") == "FAILED":
            scored_cases.append({**case, "dims": {}, "composite": 0.0, "gate_failures": ["pipeline FAILED"]})
            continue

        dims = {
            "hallucination": _dim_hallucination(case.get("hallucination_rate", 0.0)),
            "relevance":     _dim_relevance(case.get("relevance_score", 0.0)),
            "attribution":   _dim_attribution(case.get("attribution_coverage", 0.0)),
            "yield":         _dim_yield(case.get("total_claims", 0)),
            "errors":        _dim_errors(case.get("errors", 0)),
            "diversity":     _dim_diversity(case.get("unique_sources", 0)),
        }
        composite = _composite_score(dims)
        gate_failures = _check_gates({"raw": case, "dims": dims, "composite": composite}, thresholds)

        scored_cases.append({
            "company": case["company"],
            "ticker": case.get("ticker", ""),
            "status": case["status"],
            "raw": case,
            "dims": {k: round(v, 4) for k, v in dims.items()},
            "composite": composite,
            "gate_failures": gate_failures,
            "gate_status": "PASS" if not gate_failures else "FAIL",
        })

    # Aggregate across OK cases only
    ok_cases = [c for c in scored_cases if c.get("status") == "OK"]
    n = len(ok_cases)

    def _avg(key: str) -> float:
        vals = [c["dims"][key] for c in ok_cases if key in c.get("dims", {})]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_dims = {k: _avg(k) for k in _WEIGHTS}
    avg_composite = round(sum(c["composite"] for c in ok_cases) / n, 1) if n else 0.0
    grade = _letter_grade(avg_composite)

    return {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "golden_cases": eval_result.get("total_cases", len(scored_cases)),
        "passed_pipeline": eval_result.get("passed", n),
        "failed_pipeline": eval_result.get("failed", 0),
        "thresholds_used": thresholds,
        "aggregate": {
            "composite": avg_composite,
            "grade": grade,
            "dims": avg_dims,
            "latency_p50_s": eval_result.get("latency_p50_s", 0),
            "latency_p95_s": eval_result.get("latency_p95_s", 0),
            "cases_passing_gates": sum(1 for c in scored_cases if c.get("gate_status") == "PASS"),
            "cases_failing_gates": sum(1 for c in scored_cases if c.get("gate_status") == "FAIL"),
        },
        "cases": scored_cases,
        "per_agent_latency": eval_result.get("per_agent_latency", {}),
    }


# ── Historical tracking ───────────────────────────────────────────────────────

def save_result(payload: dict[str, Any]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"benchmark_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def load_latest_result() -> dict[str, Any] | None:
    if not _RESULTS_DIR.exists():
        return None
    files = sorted(_RESULTS_DIR.glob("benchmark_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _delta_str(current: float, previous: float, higher_better: bool = True) -> str:
    diff = current - previous
    symbol = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
    good = (diff > 0) == higher_better
    sign = "+" if diff > 0 else ""
    tag = " (better)" if good and diff != 0 else (" (worse)" if not good and diff != 0 else "")
    return f"{symbol} {sign}{diff:+.1f}{tag}"


# ── Score card rendering ──────────────────────────────────────────────────────

_BAR = 22  # progress bar character width


def _bar(score_0_1: float) -> str:
    filled = round(max(0.0, min(1.0, score_0_1)) * _BAR)
    return "[" + "#" * filled + "." * (_BAR - filled) + "]"


def print_scorecard(scored: dict[str, Any], previous: dict[str, Any] | None = None) -> None:
    agg = scored["aggregate"]
    cases = scored["cases"]
    sep = "=" * 74

    print(f"\n{sep}")
    print(f"  FINANCIAL ASSISTANT  |  BENCHMARK SCORE CARD")
    print(f"  Run : {scored['run_at']}")
    print(f"  Cases : {scored['golden_cases']} total  "
          f"| {scored['passed_pipeline']} OK  "
          f"| {scored['failed_pipeline']} failed")
    print(sep)

    # ── Dimension breakdown ───────────────────────────────────────────────
    print("\n  SCORING DIMENSIONS  (weight → contribution to composite)\n")

    dim_meta = [
        ("Hallucination Control", "hallucination", "30%"),
        ("Relevance (keyword hit)",  "relevance",     "25%"),
        ("Attribution Coverage",     "attribution",   "20%"),
        ("Claim Yield",              "yield",         "10%"),
        ("Error-Free Rate",          "errors",        "10%"),
        ("Source Diversity",         "diversity",      "5%"),
    ]

    prev_dims = (previous or {}).get("aggregate", {}).get("dims", {})
    for label, key, weight in dim_meta:
        score = agg["dims"].get(key, 0.0)
        bar = _bar(score)
        delta = ""
        if key in prev_dims:
            delta = "  " + _delta_str(score * 100, prev_dims[key] * 100)
        print(f"  {label:<30} {weight:>4}  {bar}  {score * 100:5.1f}/100{delta}")

    # ── Composite ─────────────────────────────────────────────────────────
    composite = agg["composite"]
    grade = agg["grade"]
    prev_composite = (previous or {}).get("aggregate", {}).get("composite")
    delta_comp = ""
    if prev_composite is not None:
        delta_comp = "  " + _delta_str(composite, prev_composite)
    print(f"\n  {'COMPOSITE SCORE':<30} {'':>4}  {_bar(composite / 100)}  "
          f"{composite:5.1f}/100  [{grade}]{delta_comp}")

    # ── Latency ───────────────────────────────────────────────────────────
    print(f"\n  Latency (financial agent)  "
          f"p50={agg['latency_p50_s']:.2f}s  p95={agg['latency_p95_s']:.2f}s")

    # ── Quality gates ─────────────────────────────────────────────────────
    print(f"\n  Quality gates  "
          f"{agg['cases_passing_gates']} PASS  /  "
          f"{agg['cases_failing_gates']} FAIL\n")

    # ── Per-case table ────────────────────────────────────────────────────
    print(f"  {'Company':<30} {'Score':>6}  {'Hall%':>6}  {'Rel%':>5}  "
          f"{'Attr%':>6}  {'Clms':>5}  {'Srcs':>5}  Gate")
    print("  " + "─" * 70)

    for c in cases:
        if c.get("status") == "FAILED":
            err = str(c.get("error", ""))[:28]
            print(f"  {c['company']:<30}  FAILED  {err}")
            continue
        raw = c.get("raw", c)
        gate = c.get("gate_status", "?")
        n_fail = len(c.get("gate_failures", []))
        gate_str = "PASS" if gate == "PASS" else f"FAIL×{n_fail}"
        print(
            f"  {c['company']:<30} {c['composite']:>6.1f}"
            f"  {raw.get('hallucination_rate', 0) * 100:>5.1f}%"
            f"  {raw.get('relevance_score', 0) * 100:>4.0f}%"
            f"  {raw.get('attribution_coverage', 0) * 100:>5.0f}%"
            f"  {raw.get('total_claims', 0):>5}"
            f"  {raw.get('unique_sources', 0):>5}"
            f"  {gate_str}"
        )

    # ── Failure details ───────────────────────────────────────────────────
    all_failures = [
        f"[{c['company']}] {f}"
        for c in cases
        for f in c.get("gate_failures", [])
    ]
    if all_failures:
        print(f"\n  QUALITY GATE FAILURES ({len(all_failures)}):")
        for line in all_failures[:15]:
            print(f"    ✗ {line}")
    else:
        print("\n  All quality gates PASSED.")

    print(f"\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_thresholds(path: Path) -> dict[str, float]:
    if path.exists():
        overrides = json.loads(path.read_text(encoding="utf-8"))
        return {**_BUILTIN_THRESHOLDS, **overrides}
    return _BUILTIN_THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Assistant Benchmark Runner")
    parser.add_argument(
        "--golden",
        default=str(_DEFAULT_GOLDEN),
        help="Path to golden_dataset.json (default: tests/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--thresholds",
        default=str(_DEFAULT_THRESHOLDS),
        help="Path to benchmark_thresholds.json overrides",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save benchmark result to tests/eval/benchmark_results/",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against the most recent saved result",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also write the scored JSON to this path",
    )
    args = parser.parse_args()

    thresholds = _load_thresholds(Path(args.thresholds))
    log.info("benchmark_start", golden=args.golden)

    # Run full pipeline evaluation
    eval_result = run_eval(args.golden)

    # Score it
    scored = score_eval_result(eval_result, thresholds)

    # Optionally load previous result for comparison
    previous: dict[str, Any] | None = None
    if args.compare:
        previous = load_latest_result()
        if previous is None:
            log.warning("benchmark_compare_no_previous", msg="No previous result found; skipping comparison")

    # Print score card
    print_scorecard(scored, previous=previous)

    # Save
    if args.save:
        saved_path = save_result(scored)
        print(f"  Result saved → {saved_path}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(scored, indent=2, default=str), encoding="utf-8")
        print(f"  Result also written → {args.output}\n")

    # Exit with non-zero if composite below minimum threshold
    if scored["aggregate"]["composite"] < thresholds["composite_score_min"]:
        log.warning(
            "benchmark_below_threshold",
            composite=scored["aggregate"]["composite"],
            minimum=thresholds["composite_score_min"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
