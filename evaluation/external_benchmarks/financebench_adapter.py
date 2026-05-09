"""
evaluation/external_benchmarks/financebench_adapter.py
--------------------------------------------------------
Adapter for the FinanceBench benchmark.

FinanceBench (Islamabad et al., 2023) is the most direct public benchmark for
financial document Q&A.  It contains ~10,000 questions over SEC 10-K / 10-Q /
earnings-call transcripts, each with a gold numeric or short-text answer.

Paper  : https://arxiv.org/abs/2311.11944
Data   : https://github.com/patronus-ai/financebench
         (Download financebench_open_source.jsonl from the Releases page)
License: Apache 2.0

HOW THIS ADAPTER WORKS
-----------------------
1. Load the JSONL file (one question per line).
2. For each question, call our pipeline's LLM (not the full agent graph –
   FinanceBench provides the source document context, so we need only the
   answering step) via get_llm_response().
3. Compare our answer to the gold answer using:
     - Exact match after normalisation (strip $, commas, %)
     - Numeric tolerance match  (±1% for numerical answers)
     - LLM judge for free-text answers
4. Emit a report with accuracy, numeric accuracy, and latency.

USAGE
-----
  # Download data first:
  #   wget https://github.com/patronus-ai/financebench/releases/download/v1.0/financebench_open_source.jsonl

  python -m evaluation.external_benchmarks.financebench_adapter \
      --data path/to/financebench_open_source.jsonl \
      --limit 100          # run on first 100 questions (full set = ~10k, slow)
      --output results/financebench_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

from llm.provider import get_llm_response, get_llm_json_response

log = structlog.get_logger(__name__)

# ── Normalisation helpers ──────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Strip currency symbols, commas, %, whitespace; lowercase."""
    text = str(text).lower().strip()
    text = re.sub(r"[$€£₹,% ]", "", text)
    return text


def _numeric(text: str) -> float | None:
    """Return float if text represents a number, else None."""
    try:
        return float(_normalise(text).replace("(", "-").replace(")", ""))
    except ValueError:
        return None


def _exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def _numeric_match(pred: str, gold: str, tol: float = 0.01) -> bool:
    """True if both are numeric and within `tol` relative tolerance."""
    pv = _numeric(pred)
    gv = _numeric(gold)
    if pv is None or gv is None:
        return False
    if gv == 0:
        return abs(pv) < 1e-6
    return abs((pv - gv) / gv) <= tol


def _llm_judge(question: str, context: str, prediction: str, gold: str) -> bool:
    """Ask the LLM whether prediction is semantically equivalent to gold."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial answer judge. "
                "Respond with JSON: {\"correct\": true} or {\"correct\": false}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Gold answer: {gold}\n"
                f"Predicted answer: {prediction}\n\n"
                "Is the predicted answer correct or equivalent to the gold answer? "
                "Allow minor formatting differences (e.g. '$1.2B' vs '1,200 million'). "
                "Return JSON."
            ),
        },
    ]
    try:
        raw = get_llm_json_response(messages, temperature=0.0, max_tokens=64)
        result = json.loads(raw) if isinstance(raw, str) else raw
        return bool(result.get("correct", False))
    except Exception:
        return False


# ── Answer generation ─────────────────────────────────────────────────────────

def _answer_question(question: str, context: str) -> str:
    """
    Call our LLM with the FinanceBench document context + question.
    Returns the raw answer string.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst. "
                "Answer the question using only the provided financial document context. "
                "Give a concise, exact answer (number + unit where applicable). "
                "Do NOT explain unless asked."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context[:4000]}\n\nQuestion: {question}",
        },
    ]
    try:
        return get_llm_response(messages, temperature=0.0, max_tokens=128)
    except Exception as exc:
        log.warning("financebench_answer_failed", error=str(exc))
        return ""


# ── Main runner ────────────────────────────────────────────────────────────────

def run_financebench(data_path: str, limit: int | None = None) -> dict[str, Any]:
    """
    Run the FinanceBench adapter.
    Returns a results dict with per-question details and aggregate metrics.
    """
    records: list[dict] = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    log.info("financebench_start", total=len(records))
    results: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        question: str = rec.get("question", "")
        gold: str = str(rec.get("answer", ""))
        context: str = rec.get("evidence", rec.get("context", ""))
        company: str = rec.get("company", "")
        doc_type: str = rec.get("doc_type", "")

        t0 = time.perf_counter()
        prediction = _answer_question(question, context)
        elapsed = time.perf_counter() - t0

        exact = _exact_match(prediction, gold)
        numeric = _numeric_match(prediction, gold)
        if not exact and not numeric:
            llm_correct = _llm_judge(question, context, prediction, gold)
        else:
            llm_correct = exact or numeric

        results.append({
            "idx": i,
            "company": company,
            "doc_type": doc_type,
            "question": question[:120],
            "gold": gold,
            "prediction": prediction[:200],
            "exact_match": exact,
            "numeric_match": numeric,
            "llm_judge_correct": llm_correct,
            "correct": exact or numeric or llm_correct,
            "elapsed_s": round(elapsed, 2),
        })
        log.info(
            "financebench_case",
            idx=i,
            correct=results[-1]["correct"],
            elapsed_s=results[-1]["elapsed_s"],
        )

    n = len(results)
    exact_acc = sum(1 for r in results if r["exact_match"]) / n if n else 0
    numeric_acc = sum(1 for r in results if r["numeric_match"]) / n if n else 0
    overall_acc = sum(1 for r in results if r["correct"]) / n if n else 0
    avg_latency = sum(r["elapsed_s"] for r in results) / n if n else 0

    summary = {
        "benchmark": "FinanceBench",
        "source": "https://github.com/patronus-ai/financebench",
        "questions_evaluated": n,
        "exact_match_accuracy": round(exact_acc, 4),
        "numeric_match_accuracy": round(numeric_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "avg_latency_s": round(avg_latency, 2),
        "per_question": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FinanceBench adapter")
    parser.add_argument("--data", required=True, help="Path to financebench_open_source.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    result = run_financebench(args.data, limit=args.limit)

    print(f"\n{'=' * 60}")
    print(f"  FinanceBench Results")
    print(f"{'=' * 60}")
    print(f"  Questions evaluated : {result['questions_evaluated']}")
    print(f"  Exact match acc.    : {result['exact_match_accuracy']:.1%}")
    print(f"  Numeric match acc.  : {result['numeric_match_accuracy']:.1%}")
    print(f"  Overall accuracy    : {result['overall_accuracy']:.1%}")
    print(f"  Avg latency/q       : {result['avg_latency_s']:.2f}s")
    print(f"{'=' * 60}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
