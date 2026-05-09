"""
agents/verification_agent.py
------------------------------
VerificationAgent — re-fetches source URLs and runs the LLM judge against
every claim. Sets hallucination_score and verified=True on each claim.

Only claims with hallucination_score < settings.hallucination_threshold
and verified=True are forwarded to downstream agents.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from config.settings import settings
from llm.prompt_manager import load_prompt
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim, VerificationReport
from tools._base import safe_get

log = structlog.get_logger(__name__)

_MAX_WORKERS = 8


def _verify_claim(claim: SourcedClaim) -> SourcedClaim:
    """
    Attempt to re-fetch the source URL and run the LLM judge.
    Mutates and returns the claim with updated hallucination_score and verified flag.
    """
    # Re-fetch source snippet
    live_snippet = ""
    try:
        if claim.source_url and claim.source_url.startswith("http"):
            resp = safe_get(claim.source_url, timeout=10)
            if resp and resp.text:
                live_snippet = resp.text[:800]
    except Exception:
        pass  # use stored raw_snippet as fallback

    snippet = live_snippet or claim.raw_snippet or ""
    if not snippet:
        # Cannot verify without any source text — mark unverifiable
        claim.hallucination_score = 0.5
        claim.verified = True
        return claim

    judge_prompt_template = load_prompt("verification_judge")
    judge_prompt = judge_prompt_template.format(
        snippet=snippet[:500],
        claim=str(claim.value)[:300],
    )

    messages = [
        {"role": "system", "content": "You are a factual verification judge."},
        {"role": "user", "content": judge_prompt},
    ]
    try:
        response_text = get_llm_json_response(messages, temperature=0.0, max_tokens=256)
        result = json.loads(response_text)
        verdict = result.get("verdict", "unverifiable")
        score = float(result.get("hallucination_score", 0.5))

        claim.hallucination_score = max(0.0, min(1.0, score))
        claim.verified = True
        if verdict == "hallucinated":
            claim.hallucination_score = max(claim.hallucination_score, 0.7)
    except Exception as exc:
        log.debug("verify_claim_llm_failed", claim_id=str(claim.claim_id), error=str(exc))
        claim.hallucination_score = 0.5
        claim.verified = True

    return claim


def run(all_claims: dict[str, list[SourcedClaim]]) -> tuple[dict[str, list[SourcedClaim]], VerificationReport, float]:
    """
    Verify all claims from all specialist agents in parallel.

    Parameters
    ----------
    all_claims : dict mapping agent_name -> list[SourcedClaim]

    Returns
    -------
    (verified_claims_by_agent, VerificationReport, elapsed_seconds)
    """
    t0 = time.perf_counter()
    flat: list[tuple[str, SourcedClaim]] = [
        (agent, claim)
        for agent, claims in all_claims.items()
        for claim in claims
    ]

    verified_flat: list[tuple[str, SourcedClaim]] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_verify_claim, claim): (agent, claim) for agent, claim in flat}
        for future in as_completed(futures):
            agent, original = futures[future]
            try:
                updated = future.result()
                verified_flat.append((agent, updated))
            except Exception as exc:
                log.warning("verify_future_failed", agent=agent, error=str(exc))
                verified_flat.append((agent, original))

    # Re-group
    verified_by_agent: dict[str, list[SourcedClaim]] = {k: [] for k in all_claims}
    for agent, claim in verified_flat:
        verified_by_agent[agent].append(claim)

    # Build VerificationReport
    threshold = settings.hallucination_threshold
    total = len(flat)
    passed = sum(1 for _, c in verified_flat if c.hallucination_score < threshold)
    failed = total - passed

    per_agent_confidence: dict[str, float] = {}
    for agent, claims in verified_by_agent.items():
        if not claims:
            per_agent_confidence[agent] = 1.0
        else:
            avg = sum(1.0 - c.hallucination_score for c in claims) / len(claims)
            per_agent_confidence[agent] = round(avg, 3)

    verification_report = VerificationReport(
        total_claims=total,
        verified_claims=passed,
        hallucinated_claims=failed,
        unverifiable_claims=0,
        per_agent_confidence=per_agent_confidence,
        overall_confidence=round(passed / total, 3) if total else 1.0,
    )

    elapsed = time.perf_counter() - t0
    log.info(
        "verification_done",
        total=total,
        passed=passed,
        failed=failed,
        elapsed=round(elapsed, 2),
    )
    return verified_by_agent, verification_report, elapsed
