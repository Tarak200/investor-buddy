"""
evaluation/external_benchmarks/compare_all.py
----------------------------------------------
Unified comparison table showing our system's scores on all external benchmarks
alongside published numbers from the literature.

Published reference numbers (as of May 2026)
----------------------------------------------
Benchmark          Metric           GPT-4  FinBERT  BloombergGPT  FinGPT
─────────────────  ───────────────  ─────  ───────  ────────────  ──────
FinanceBench       Overall Acc.     ~81%   –        –             –
FinQA              Numeric Acc.     68.0%  –        43.4%         –
FPB (allAgree)     Weighted F1      ~87%   88.8%    –             75.1%
FiQA-SA            Pearson r        ~0.80  –        –             0.75
RAGAS Faithfulness (varies by RAG setup)

Sources:
  FinanceBench : https://arxiv.org/abs/2311.11944
  FinQA        : https://arxiv.org/abs/2109.00122
  FPB          : https://arxiv.org/abs/2306.11944 (FLARE paper)
  BloombergGPT : https://arxiv.org/abs/2303.17564

USAGE
-----
  # After running all adapters and saving their output:
  python -m evaluation.external_benchmarks.compare_all \
      --financebench results/financebench_results.json \
      --finqa        results/finqa_results.json \
      --fpb          results/fpb_results.json \
      --fiqasa       results/fiqasa_results.json \
      --ragas        results/ragas_results.json

  Any argument can be omitted if that benchmark hasn't been run yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ── Published reference scores ────────────────────────────────────────────────

_REFERENCE = {
    "financebench_overall_acc": {
        "GPT-4o":         0.810,
        "GPT-4":          0.790,
        "Claude-3-Opus":  0.760,
    },
    "finqa_numeric_acc": {
        "GPT-4":          0.680,
        "BloombergGPT":   0.434,
        "FinMA-30B":      0.490,
    },
    "fpb_weighted_f1": {
        "GPT-4 (0-shot)": 0.870,
        "FinBERT":        0.888,
        "BloombergGPT":   0.510,   # reported in FLARE paper
        "FinGPT-v3":      0.751,
    },
    "fiqasa_pearson": {
        "GPT-4 (0-shot)": 0.800,
        "FinGPT-v3":      0.750,
        "BloombergGPT":   0.680,
    },
}


def _load(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _bar(score: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, score)) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def _ref_row(key: str) -> str:
    refs = _REFERENCE.get(key, {})
    if not refs:
        return ""
    parts = [f"{name}: {_pct(score)}" for name, score in refs.items()]
    return "  Reference: " + "  |  ".join(parts)


def print_comparison(
    financebench: dict | None,
    finqa: dict | None,
    fpb: dict | None,
    fiqasa: dict | None,
    ragas: dict | None,
) -> None:
    sep = "=" * 74

    print(f"\n{sep}")
    print(f"  FINANCIAL ASSISTANT vs. PUBLIC BENCHMARKS")
    print(sep)

    # ── FinanceBench ─────────────────────────────────────────────────────────
    print(f"\n  1. FINANCEBENCH  (financial doc Q&A accuracy)")
    if financebench:
        acc = financebench.get("overall_accuracy", 0)
        n = financebench.get("questions_evaluated", 0)
        print(f"     Questions : {n}")
        print(f"     Overall Acc.  {_bar(acc)} {_pct(acc)}")
        print(f"     Exact Match   {_bar(financebench.get('exact_match_accuracy', 0))} {_pct(financebench.get('exact_match_accuracy', 0))}")
        print(f"     Numeric Match {_bar(financebench.get('numeric_match_accuracy', 0))} {_pct(financebench.get('numeric_match_accuracy', 0))}")
        print(_ref_row("financebench_overall_acc"))
    else:
        print("     Not run yet. Run: python -m evaluation.external_benchmarks.financebench_adapter --data <file>")

    # ── FinQA ────────────────────────────────────────────────────────────────
    print(f"\n  2. FINQA  (multi-step numerical reasoning)")
    if finqa:
        acc = finqa.get("numeric_accuracy", 0)
        n = finqa.get("questions_evaluated", 0)
        print(f"     Questions : {n}")
        print(f"     Numeric Acc.  {_bar(acc)} {_pct(acc)}")
        print(f"     Answer Rate   {_bar(finqa.get('answered_pct', 0))} {_pct(finqa.get('answered_pct', 0))}")
        print(_ref_row("finqa_numeric_acc"))
    else:
        print("     Not run yet. Run: python -m evaluation.external_benchmarks.finqa_adapter --data <file>")

    # ── FPB ──────────────────────────────────────────────────────────────────
    print(f"\n  3. FINANCIAL PHRASEBANK  (sentiment classification)")
    if fpb:
        f1 = fpb.get("weighted_f1", 0)
        acc = fpb.get("accuracy", 0)
        n = fpb.get("questions_evaluated", 0)
        print(f"     Sentences : {n}")
        print(f"     Weighted F1  {_bar(f1)} {f1:.4f}")
        print(f"     Accuracy     {_bar(acc)} {_pct(acc)}")
        for lbl, score in fpb.get("f1_per_class", {}).items():
            print(f"       {lbl:<12} F1 = {score:.4f}")
        print(_ref_row("fpb_weighted_f1"))
    else:
        print("     Not run yet. Run: python -m evaluation.external_benchmarks.flare_adapter --task fpb")

    # ── FiQA-SA ──────────────────────────────────────────────────────────────
    print(f"\n  4. FLARE FiQA-SA  (continuous sentiment scoring)")
    if fiqasa:
        pearson = fiqasa.get("pearson_correlation", 0)
        mae = fiqasa.get("avg_mae", 0)
        n = fiqasa.get("questions_evaluated", 0)
        print(f"     Sentences    : {n}")
        print(f"     Pearson r    : {pearson:.4f}  (higher = better)")
        print(f"     Avg MAE      : {mae:.4f}  (lower = better)")
        print(_ref_row("fiqasa_pearson"))
    else:
        print("     Not run yet. Run: python -m evaluation.external_benchmarks.flare_adapter --task fiqasa --data <file>")

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    print(f"\n  5. RAGAS  (RAG pipeline quality – no external download needed)")
    if ragas:
        agg = ragas.get("aggregate", {})
        print(f"     Cases evaluated  : {ragas.get('cases_evaluated', 0)}")
        print(f"     Faithfulness     {_bar(agg.get('faithfulness', 0))} {agg.get('faithfulness', 0):.4f}  (1 − hallucination rate)")
        print(f"     Answer Relevance {_bar(agg.get('answer_relevance', 0))} {agg.get('answer_relevance', 0):.4f}  (keyword hit rate)")
        print(f"     Context Prec.    {_bar(agg.get('context_precision', 0))} {agg.get('context_precision', 0):.4f}  (attribution coverage)")
        print(f"     Context Recall   {_bar(agg.get('context_recall', 0))} {agg.get('context_recall', 0):.4f}  (keywords in sources)")
        rs = agg.get("ragas_score", 0)
        print(f"     ── RAGAS Score   {_bar(rs)} {rs:.4f}  (mean of 4 metrics)")
    else:
        print("     Not run yet. Run: python -m evaluation.external_benchmarks.ragas_adapter --results <saved_benchmark.json>")

    print(f"\n{sep}")
    print(f"  QUICK-START: run all benchmarks from the project root")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  # 1. Internal benchmark (no external data)")
    print(f"  python -m evaluation.benchmark --save")
    print()
    print(f"  # 2. RAGAS (uses saved benchmark result, no extra data)")
    print(f"  python -m evaluation.external_benchmarks.ragas_adapter \\")
    print(f"      --results tests/eval/benchmark_results/<latest>.json")
    print()
    print(f"  # 3. Financial PhraseBank (auto-downloads from HuggingFace)")
    print(f"  pip install datasets")
    print(f"  python -m evaluation.external_benchmarks.flare_adapter --task fpb --limit 500")
    print()
    print(f"  # 4. FinanceBench (download JSONL from GitHub releases)")
    print(f"  python -m evaluation.external_benchmarks.financebench_adapter \\")
    print(f"      --data financebench_open_source.jsonl --limit 100")
    print()
    print(f"  # 5. FinQA (clone repo from GitHub)")
    print(f"  python -m evaluation.external_benchmarks.finqa_adapter \\")
    print(f"      --data /path/to/FinQA/dataset/test.json --limit 50")
    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare our system against public benchmarks")
    parser.add_argument("--financebench", default=None)
    parser.add_argument("--finqa", default=None)
    parser.add_argument("--fpb", default=None)
    parser.add_argument("--fiqasa", default=None)
    parser.add_argument("--ragas", default=None)
    args = parser.parse_args()

    print_comparison(
        financebench=_load(args.financebench),
        finqa=_load(args.finqa),
        fpb=_load(args.fpb),
        fiqasa=_load(args.fiqasa),
        ragas=_load(args.ragas),
    )


if __name__ == "__main__":
    main()
