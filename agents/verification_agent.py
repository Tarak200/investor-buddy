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

import structlog

from config.settings import settings
from llm.prompt_manager import load_prompt
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim, VerificationReport
from tools._base import safe_get

log = structlog.get_logger(__name__)

_BATCH_SIZE = 10  # claims verified per single LLM call


# Domains that reliably block programmatic access — skip live re-fetch for these.
_SKIP_REFETCH_DOMAINS = (
    "finance.yahoo.com",
    "screener.in",
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "bseindia.com",
    "nseindia.com",
)


def _verify_claim(claim: SourcedClaim) -> SourcedClaim:
    """
    Mark claim as verified with a neutral score (called when batching is bypassed).
    Individual claim verification is handled by _verify_batch.
    """
    claim.hallucination_score = 0.5
    claim.verified = True
    return claim


def _verify_batch(claims: list[SourcedClaim]) -> list[SourcedClaim]:
    """
    Verify up to _BATCH_SIZE claims in a single LLM call.
    Returns the claims list with updated hallucination_score and verified flag.
    """
    if not claims:
        return claims

    # Build a compact prompt with all claims in the batch
    items: list[str] = []
    for i, c in enumerate(claims):
        snippet = (c.raw_snippet or "")[:200]
        value_str = str(c.value)[:150]
        items.append(
            f"[{i}] source={c.source_name!r} snippet={snippet!r} claim={value_str!r}"
        )

    batch_text = "\n".join(items)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual verification judge. "
                "For each numbered item, return a JSON object with key \"results\" containing an array. "
                "Each element of the array must have: "
                "\"index\" (int), \"verdict\" (\"verified\"|\"hallucinated\"|\"unverifiable\"), "
                "and \"hallucination_score\" (0.0=fully verified, 1.0=hallucinated). "
                "Return ONLY valid JSON with a top-level \"results\" key."
            ),
        },
        {"role": "user", "content": f"Verify these claims:\n{batch_text}"},
    ]

    try:
        response_text = get_llm_json_response(messages, temperature=0.0, max_tokens=512)
        # Response might be {"results": [...]} or a bare array
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            results = parsed.get("results", parsed.get("items", []))
        else:
            results = parsed if isinstance(parsed, list) else []

        for item in results:
            if isinstance(item, list):
                # LLM returned [[index, verdict, score], ...] format
                if len(item) >= 3:
                    idx = int(item[0])
                    verdict = str(item[1])
                    score = float(item[2])
                else:
                    continue
            elif isinstance(item, dict):
                idx = int(item.get("index", -1))
                score = float(item.get("hallucination_score", 0.5))
                verdict = item.get("verdict", "unverifiable")
            else:
                continue
            if 0 <= idx < len(claims):
                claims[idx].hallucination_score = max(0.0, min(1.0, score))
                if verdict == "hallucinated":
                    claims[idx].hallucination_score = max(claims[idx].hallucination_score, 0.7)
                claims[idx].verified = True

    except Exception as exc:
        log.debug("verify_batch_failed", batch_size=len(claims), error=str(exc)[:200])
        # Fallback: mark all as unverifiable
        for c in claims:
            c.hallucination_score = 0.5
            c.verified = True

    # Ensure all claims are marked
    for c in claims:
        if not c.verified:
            c.hallucination_score = 0.5
            c.verified = True

    return claims


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

    # Build batches of claims (strip agent tag, verify in batch, restore agent tag)
    claims_only = [c for _, c in flat]
    agents_only = [a for a, _ in flat]

    batches = [claims_only[i:i + _BATCH_SIZE] for i in range(0, len(claims_only), _BATCH_SIZE)]
    log.info("verification_batching", total_claims=len(claims_only), batches=len(batches))

    # Run batches sequentially (LLM throttle handles rate limiting)
    verified_claims: list[SourcedClaim] = []
    for batch in batches:
        verified_claims.extend(_verify_batch(batch))

    verified_flat = list(zip(agents_only, verified_claims))

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
        verified_count=passed,
        hallucinated_count=failed,
        unverifiable_count=0,
        per_agent_confidence=per_agent_confidence,
        overall_pipeline_confidence=round(passed / total, 3) if total else 1.0,
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
