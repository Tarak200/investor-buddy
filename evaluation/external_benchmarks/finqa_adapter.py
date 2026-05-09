"""
evaluation/external_benchmarks/finqa_adapter.py
-------------------------------------------------
Adapter for the FinQA benchmark.

FinQA (Chen et al., 2021) tests multi-step numerical reasoning over financial
reports (SEC 10-K / 10-Q).  Each sample has a long financial document context
(tables + text) and requires a program/chain-of-thought answer using arithmetic
operators (add, subtract, multiply, divide, table-lookup).

Paper  : https://arxiv.org/abs/2109.00122
Data   : https://github.com/czyssrs/FinQA  (data/ folder, JSON format)
License: MIT

HOW THIS ADAPTER WORKS
-----------------------
1. Load train.json / test.json from the FinQA repo.
2. For each sample, reconstruct the context from the pre/post text + table.
3. Ask our LLM to answer with a step-by-step chain-of-thought.
4. Extract the final numeric answer from the LLM response.
5. Compare to the gold answer using numeric tolerance (±1%).
6. Report: exact program match, numeric accuracy, execution accuracy.

FinQA context structure per sample:
  {
    "id": "...",
    "pre_text": [...],     # paragraphs before the table
    "post_text": [...],    # paragraphs after the table
    "table": [[...]],      # 2-D list (headers + rows)
    "qa": {
      "question": "...",
      "answer": "...",     # e.g. "5.4%" or "1234.5"
      "exe_ans": 5.4       # pre-computed float answer
    }
  }

USAGE
-----
  # Clone FinQA repo and point to the test set:
  #   git clone https://github.com/czyssrs/FinQA /tmp/FinQA

  python -m evaluation.external_benchmarks.finqa_adapter \
      --data /tmp/FinQA/dataset/test.json \
      --limit 50 \
      --output results/finqa_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

from llm.provider import get_llm_response

log = structlog.get_logger(__name__)


# ── Context assembly ───────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list[str]]) -> str:
    """Convert a 2-D list into a simple Markdown table string."""
    if not table:
        return ""
    header = " | ".join(str(c) for c in table[0])
    sep = " | ".join("---" for _ in table[0])
    rows = "\n".join(" | ".join(str(c) for c in row) for row in table[1:])
    return f"{header}\n{sep}\n{rows}"


def _build_context(sample: dict) -> str:
    pre = " ".join(sample.get("pre_text", []))
    post = " ".join(sample.get("post_text", []))
    table_md = _table_to_markdown(sample.get("table", []))
    return f"{pre}\n\nTABLE:\n{table_md}\n\n{post}"


# ── Answer extraction ─────────────────────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    """Pull the last numeric value from the LLM response."""
    # Match numbers like 1,234.5 or -5.4% or (123) for negative
    matches = re.findall(r"-?\(?\d[\d,]*\.?\d*%?\)?", text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _numeric_match(pred: float | None, gold: float, tol: float = 0.01) -> bool:
    if pred is None:
        return False
    if gold == 0:
        return abs(pred) < 1e-6
    return abs((pred - gold) / gold) <= tol


# ── Answer generation ─────────────────────────────────────────────────────────

def _answer_finqa(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst performing numerical reasoning over financial statements. "
                "Show your step-by-step calculation, then state the final answer on the last line "
                "in the format: ANSWER: <number>"
            ),
        },
        {
            "role": "user",
            "content": f"Financial document:\n{context[:5000]}\n\nQuestion: {question}",
        },
    ]
    try:
        return get_llm_response(messages, temperature=0.0, max_tokens=512)
    except Exception as exc:
        log.warning("finqa_answer_failed", error=str(exc))
        return ""


# ── Main runner ────────────────────────────────────────────────────────────────

def run_finqa(data_path: str, limit: int | None = None) -> dict[str, Any]:
    samples: list[dict] = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if limit:
        samples = samples[:limit]

    log.info("finqa_start", total=len(samples))
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        qa = sample.get("qa", {})
        question: str = qa.get("question", "")
        gold_str: str = str(qa.get("answer", ""))
        gold_float: float = float(qa.get("exe_ans", 0.0))
        context = _build_context(sample)

        t0 = time.perf_counter()
        response = _answer_finqa(question, context)
        elapsed = time.perf_counter() - t0

        pred_num = _extract_number(response)
        correct = _numeric_match(pred_num, gold_float)

        results.append({
            "idx": i,
            "id": sample.get("id", ""),
            "question": question[:120],
            "gold_str": gold_str,
            "gold_float": gold_float,
            "predicted_num": pred_num,
            "correct": correct,
            "response_excerpt": response[:300],
            "elapsed_s": round(elapsed, 2),
        })
        log.info("finqa_case", idx=i, correct=correct, pred=pred_num, gold=gold_float)

    n = len(results)
    accuracy = sum(1 for r in results if r["correct"]) / n if n else 0
    avg_latency = sum(r["elapsed_s"] for r in results) / n if n else 0

    # Break down accuracy by answered vs unanswered
    answered = [r for r in results if r["predicted_num"] is not None]
    answered_acc = sum(1 for r in answered if r["correct"]) / len(answered) if answered else 0

    return {
        "benchmark": "FinQA",
        "source": "https://github.com/czyssrs/FinQA",
        "questions_evaluated": n,
        "numeric_accuracy": round(accuracy, 4),
        "answered_pct": round(len(answered) / n, 4) if n else 0,
        "answered_numeric_accuracy": round(answered_acc, 4),
        "avg_latency_s": round(avg_latency, 2),
        "per_question": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FinQA adapter")
    parser.add_argument("--data", required=True, help="Path to FinQA test.json or train.json")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    result = run_finqa(args.data, limit=args.limit)

    print(f"\n{'=' * 60}")
    print(f"  FinQA Results  (numerical reasoning)")
    print(f"{'=' * 60}")
    print(f"  Questions evaluated     : {result['questions_evaluated']}")
    print(f"  Numeric accuracy        : {result['numeric_accuracy']:.1%}")
    print(f"  Answer extraction rate  : {result['answered_pct']:.1%}")
    print(f"  Accuracy (when answered): {result['answered_numeric_accuracy']:.1%}")
    print(f"  Avg latency/q           : {result['avg_latency_s']:.2f}s")
    print(f"{'=' * 60}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
