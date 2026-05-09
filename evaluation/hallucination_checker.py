"""
evaluation/hallucination_checker.py
-------------------------------------
Standalone hallucination checker utility.  Used by the VerificationAgent
and the evaluation harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import structlog

from llm.prompt_manager import load_prompt
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)

Verdict = Literal["factual", "hallucinated", "unverifiable"]


@dataclass
class CheckResult:
    verdict: Verdict
    hallucination_score: float  # 0.0 = definitely factual, 1.0 = definitely hallucinated
    explanation: str


def check_claim(claim: SourcedClaim, snippet: str | None = None) -> CheckResult:
    """
    Run the LLM judge on a single claim.
    `snippet` — live re-fetched source text (or None to use claim.raw_snippet).
    """
    source_text = (snippet or claim.raw_snippet or "")[:500]
    if not source_text:
        return CheckResult(verdict="unverifiable", hallucination_score=0.5, explanation="No source text available")

    judge_template = load_prompt("verification_judge")
    prompt = judge_template.format(
        snippet=source_text,
        claim=str(claim.value)[:300],
    )
    messages = [
        {"role": "system", "content": "You are a factual verification judge."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = get_llm_json_response(messages, temperature=0.0, max_tokens=256)
        result = json.loads(response)
        verdict: Verdict = result.get("verdict", "unverifiable")
        score = float(result.get("hallucination_score", 0.5))
        score = max(0.0, min(1.0, score))
        return CheckResult(
            verdict=verdict,
            hallucination_score=score,
            explanation=str(result.get("explanation", "")),
        )
    except Exception as exc:
        log.warning("hallucination_check_failed", error=str(exc))
        return CheckResult(verdict="unverifiable", hallucination_score=0.5, explanation=str(exc))


def check_claims_batch(claims: list[SourcedClaim]) -> list[CheckResult]:
    """Run check_claim on each claim in the list."""
    return [check_claim(c) for c in claims]


def flag_attribution_gaps(claims: list[SourcedClaim]) -> list[str]:
    """
    Return a list of claim IDs where source_url is missing or raw_snippet is empty.
    These are structural attribution gaps that cannot be verified regardless of LLM verdict.
    """
    gaps: list[str] = []
    for c in claims:
        if not c.source_url or not c.raw_snippet:
            gaps.append(str(c.claim_id))
    return gaps
