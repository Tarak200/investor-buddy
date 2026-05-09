"""
evaluation/external_benchmarks/flare_adapter.py
-------------------------------------------------
Adapter for the FLARE benchmark suite and the Financial PhraseBank (FPB).

TWO TASKS ARE COVERED HERE:

────────────────────────────────────────────────────────────────────────────────
TASK A: Financial PhraseBank (FPB) -- Sentiment Classification
────────────────────────────────────────────────────────────────────────────────
  A classic 3-class (positive / negative / neutral) sentiment classification
  benchmark of 4,845 financial news sentences, annotated by domain experts.

  Paper  : Malo et al. (2014) – "Good Debt or Bad Debt: Detecting Semantic
           Orientations in Economic Texts"
           https://dl.acm.org/doi/10.1145/2535526
  Data   : https://huggingface.co/datasets/financial_phrasebank
           (four agreement splits: 50agree, 66agree, 75agree, allAgree)
  Metric : Weighted F1 + accuracy (matches most published numbers)

────────────────────────────────────────────────────────────────────────────────
TASK B: FLARE-FiQA-SA -- Aspect-based Sentiment (FiQA subset)
────────────────────────────────────────────────────────────────────────────────
  The FLARE benchmark (Yang et al., 2023) unifies 9 financial NLP tasks.
  The FiQA-SA subset tests sentiment on financial microblogs / news headlines
  with a continuous score (-1 to +1).

  Paper  : https://arxiv.org/abs/2306.11944
  Data   : https://github.com/the-flare/FLARE  (tasks/FiQA-SA/)
  Metric : Pearson correlation, MAE

HOW THIS ADAPTER WORKS
-----------------------
For both tasks:
  1. Load the dataset (JSONL or CSV).
  2. For each sentence, call our LLM to classify or score it.
  3. Compare to gold label with the standard metric.
  4. Report results.

USAGE (FPB)
-----------
  # Using Hugging Face datasets library:
  pip install datasets

  python -m evaluation.external_benchmarks.flare_adapter \
      --task fpb \
      --limit 200 \
      --output results/fpb_results.json

USAGE (FiQA-SA from local FLARE clone)
---------------------------------------
  git clone https://github.com/the-flare/FLARE /tmp/FLARE

  python -m evaluation.external_benchmarks.flare_adapter \
      --task fiqasa \
      --data /tmp/FLARE/tasks/FiQA-SA/test.jsonl \
      --limit 100 \
      --output results/fiqasa_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import structlog

from llm.provider import get_llm_response, get_llm_json_response

log = structlog.get_logger(__name__)

# ── Label normalisation ───────────────────────────────────────────────────────

_LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}
_REVERSE_MAP = {1: "positive", -1: "negative", 0: "neutral"}


def _parse_sentiment_label(text: str) -> str:
    """Extract 'positive' / 'negative' / 'neutral' from LLM output."""
    text_lower = text.lower()
    if "positive" in text_lower:
        return "positive"
    if "negative" in text_lower:
        return "negative"
    return "neutral"


def _parse_sentiment_score(text: str) -> float:
    """Extract a float between -1 and +1 from LLM output."""
    import re
    matches = re.findall(r"-?\d+\.?\d*", text)
    for m in reversed(matches):
        try:
            val = float(m)
            if -1.0 <= val <= 1.0:
                return val
        except ValueError:
            continue
    # Fallback: coerce label to score
    label = _parse_sentiment_label(text)
    return float(_LABEL_MAP.get(label, 0))


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _classify_fpb(sentence: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial sentiment classifier. "
                "Classify the sentiment of the following financial news sentence. "
                "Respond with ONLY one word: positive, negative, or neutral."
            ),
        },
        {"role": "user", "content": sentence},
    ]
    try:
        return _parse_sentiment_label(get_llm_response(messages, temperature=0.0, max_tokens=10))
    except Exception as exc:
        log.warning("fpb_classify_failed", error=str(exc))
        return "neutral"


def _score_fiqasa(sentence: str) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial sentiment scorer. "
                "Score the sentiment of the following financial headline or microblog post "
                "on a continuous scale from -1.0 (very negative) to +1.0 (very positive). "
                "Respond with ONLY a number between -1.0 and 1.0."
            ),
        },
        {"role": "user", "content": sentence},
    ]
    try:
        return _parse_sentiment_score(get_llm_response(messages, temperature=0.0, max_tokens=10))
    except Exception as exc:
        log.warning("fiqasa_score_failed", error=str(exc))
        return 0.0


# ── FPB runner ────────────────────────────────────────────────────────────────

def run_fpb(limit: int | None = None) -> dict[str, Any]:
    """
    Run Financial PhraseBank evaluation using the Hugging Face datasets library.
    Uses the 'allAgree' split (highest quality annotations).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "Install the datasets library first: pip install datasets"
        )

    ds = load_dataset("financial_phrasebank", "sentences_allAgree", split="train")
    # FPB has no official test split; we evaluate on the full dataset
    # (standard practice in the literature when comparing zero-shot models)
    label_names = ["negative", "neutral", "positive"]

    samples = list(ds)
    if limit:
        samples = samples[:limit]

    log.info("fpb_start", total=len(samples))
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        sentence: str = sample["sentence"]
        gold_idx: int = sample["label"]
        gold_label: str = label_names[gold_idx]

        t0 = time.perf_counter()
        pred_label = _classify_fpb(sentence)
        elapsed = time.perf_counter() - t0

        correct = pred_label == gold_label
        results.append({
            "idx": i,
            "sentence": sentence[:120],
            "gold": gold_label,
            "predicted": pred_label,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
        })
        log.info("fpb_case", idx=i, correct=correct, gold=gold_label, pred=pred_label)

    n = len(results)
    accuracy = sum(1 for r in results if r["correct"]) / n if n else 0

    # Per-class metrics for weighted F1
    from collections import Counter, defaultdict
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    label_counts: Counter = Counter(r["gold"] for r in results)

    for r in results:
        g, p = r["gold"], r["predicted"]
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    f1_per_class: dict[str, float] = {}
    for lbl in label_names:
        precision = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0
        recall = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0
        f1_per_class[lbl] = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    total = n
    weighted_f1 = (
        sum(f1_per_class[lbl] * label_counts[lbl] for lbl in label_names) / total
        if total
        else 0
    )

    return {
        "benchmark": "Financial PhraseBank (allAgree split)",
        "source": "https://huggingface.co/datasets/financial_phrasebank",
        "questions_evaluated": n,
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4),
        "f1_per_class": {k: round(v, 4) for k, v in f1_per_class.items()},
        "per_sentence": results,
    }


# ── FiQA-SA runner ────────────────────────────────────────────────────────────

def run_fiqasa(data_path: str, limit: int | None = None) -> dict[str, Any]:
    """Run FLARE FiQA-SA evaluation from a local JSONL file."""
    records: list[dict] = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    log.info("fiqasa_start", total=len(records))
    results: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        sentence: str = rec.get("sentence", rec.get("text", ""))
        gold_score: float = float(rec.get("sentiment_score", rec.get("label", 0.0)))

        t0 = time.perf_counter()
        pred_score = _score_fiqasa(sentence)
        elapsed = time.perf_counter() - t0

        mae = abs(pred_score - gold_score)
        results.append({
            "idx": i,
            "sentence": sentence[:120],
            "gold_score": gold_score,
            "predicted_score": pred_score,
            "mae": round(mae, 4),
            "elapsed_s": round(elapsed, 2),
        })
        log.info("fiqasa_case", idx=i, gold=gold_score, pred=pred_score, mae=mae)

    n = len(results)
    avg_mae = sum(r["mae"] for r in results) / n if n else 0

    # Pearson correlation
    import math
    golds = [r["gold_score"] for r in results]
    preds = [r["predicted_score"] for r in results]
    g_mean = sum(golds) / n if n else 0
    p_mean = sum(preds) / n if n else 0
    cov = sum((g - g_mean) * (p - p_mean) for g, p in zip(golds, preds)) / n if n else 0
    g_std = math.sqrt(sum((g - g_mean) ** 2 for g in golds) / n) if n else 1
    p_std = math.sqrt(sum((p - p_mean) ** 2 for p in preds) / n) if n else 1
    pearson = cov / (g_std * p_std) if (g_std * p_std) > 0 else 0

    return {
        "benchmark": "FLARE FiQA-SA",
        "source": "https://github.com/the-flare/FLARE",
        "questions_evaluated": n,
        "avg_mae": round(avg_mae, 4),
        "pearson_correlation": round(pearson, 4),
        "per_sentence": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FLARE / FPB adapter")
    parser.add_argument(
        "--task",
        choices=["fpb", "fiqasa"],
        required=True,
        help="fpb = Financial PhraseBank | fiqasa = FLARE FiQA-SA",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to JSONL data file (required for fiqasa; fpb loads from HuggingFace)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.task == "fpb":
        result = run_fpb(limit=args.limit)
        print(f"\n{'=' * 60}")
        print(f"  Financial PhraseBank Results")
        print(f"{'=' * 60}")
        print(f"  Sentences evaluated : {result['questions_evaluated']}")
        print(f"  Accuracy            : {result['accuracy']:.1%}")
        print(f"  Weighted F1         : {result['weighted_f1']:.4f}")
        for lbl, f1 in result["f1_per_class"].items():
            print(f"    {lbl:<12}: F1 = {f1:.4f}")
        print(f"{'=' * 60}\n")
    else:
        if not args.data:
            parser.error("--data is required for fiqasa task")
        result = run_fiqasa(args.data, limit=args.limit)
        print(f"\n{'=' * 60}")
        print(f"  FLARE FiQA-SA Results")
        print(f"{'=' * 60}")
        print(f"  Sentences evaluated : {result['questions_evaluated']}")
        print(f"  Avg MAE             : {result['avg_mae']:.4f}")
        print(f"  Pearson r           : {result['pearson_correlation']:.4f}")
        print(f"{'=' * 60}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
