"""
agents/discovery_agent.py
--------------------------
DiscoveryAgent — the "interesting new stocks" pipeline.

Workflow
--------
1. Ask the user for market preference (US or INDIA) — handled at the API layer,
   this agent receives `market` as a parameter.
2. Run SuperstarAgent + GovtSchemeAgent in parallel (ThreadPoolExecutor).
3. Parse all claims to extract candidate stock tickers + companies.
4. De-duplicate and score candidates (superstar conviction × policy tailwind).
5. For each top candidate (up to MAX_CANDIDATES), run a lightweight 3-agent
   deep-dive: financial, valuation, news.
6. Score and rank the candidates; return the top N recommendations with full
   rationale.

Returns: DiscoveryResult dataclass (see models below).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any

import structlog

from config.settings import settings
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)

# ── tuning constants (driven by .env / config/settings.py) ────────────────────
MAX_CANDIDATES    = settings.discovery_max_candidates
TOP_PICKS         = settings.discovery_top_picks
DEEP_DIVE_WORKERS = settings.discovery_deep_dive_workers


# ── result models ──────────────────────────────────────────────────────────────

@dataclass
class StockCandidate:
    ticker: str
    company: str
    sector: str
    market: str
    superstar_conviction: float    # 0–1; how many superstars / institutions picked it
    policy_tailwind: float         # 0–1; government scheme benefit score
    financial_score: float = 0.0  # filled by deep-dive
    valuation_score: float = 0.0  # filled by deep-dive
    news_sentiment: float = 0.0   # filled by deep-dive
    composite_score: float = 0.0  # computed after deep-dive
    investors_backing: list[str] = field(default_factory=list)
    policy_catalysts: list[str]   = field(default_factory=list)
    rationale: str = ""
    deep_dive_claims: list[SourcedClaim] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    market: str
    elapsed_seconds: float
    candidates_evaluated: int
    top_picks: list[StockCandidate]
    all_superstar_claims: list[SourcedClaim]
    all_scheme_claims: list[SourcedClaim]
    errors: list[str]


# ── candidate extraction ───────────────────────────────────────────────────────

def _extract_candidates_from_claims(
    claims: list[SourcedClaim],
    market: str,
) -> list[dict[str, Any]]:
    """
    Parse SourcedClaim values to extract structured candidate stock dicts.
    Looks for keys: new_picks, all_india_new_picks, all_us_new_picks,
    institutional_picks, policy_picks, ira_chips_beneficiaries.
    """
    raw_candidates: list[dict] = []

    for claim in claims:
        val = claim.value
        if not isinstance(val, dict):
            continue

        for key in (
            "new_picks",
            "all_india_new_picks",
            "all_us_new_picks",
            "institutional_picks",
            "policy_picks",
            "ira_chips_beneficiaries",
            "budget_schemes",
            "india_schemes",
            "us_schemes",
        ):
            items = val.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("ticker"):
                        raw_candidates.append({
                            "ticker":    item.get("ticker", "").upper().strip(),
                            "company":   item.get("company", ""),
                            "sector":    item.get("sector", "General"),
                            "investor":  item.get("investor", item.get("institution", "")),
                            "policy":    item.get("policy_catalyst",
                                         item.get("benefit_description",
                                         item.get("rationale", ""))),
                            "action":    item.get("action", ""),
                            "source_key": key,
                            "confidence": float(claim.confidence),
                        })

    return raw_candidates


def _consolidate_candidates(
    raw: list[dict[str, Any]],
    market: str,
) -> list[StockCandidate]:
    """
    Merge duplicate tickers, compute superstar_conviction and policy_tailwind.
    """
    by_ticker: dict[str, dict] = {}
    for item in raw:
        ticker = item["ticker"]
        if not ticker:
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = {
                "ticker":    ticker,
                "company":   item["company"],
                "sector":    item["sector"],
                "market":    market.upper(),
                "investors": [],
                "policies":  [],
                "superstar_hits": 0,
                "policy_hits": 0,
                "confidence_sum": 0.0,
                "count": 0,
            }
        entry = by_ticker[ticker]
        # Keep best company name
        if item["company"] and not entry["company"]:
            entry["company"] = item["company"]
        if item["investor"]:
            entry["investors"].append(item["investor"])
        if item["policy"]:
            entry["policies"].append(item["policy"])
        # Tag pick type
        sk = item.get("source_key", "")
        if "picks" in sk or "institutional" in sk:
            entry["superstar_hits"] += 1
        if "policy" in sk or "scheme" in sk or "ira" in sk or "chips" in sk:
            entry["policy_hits"] += 1
        entry["confidence_sum"] += item["confidence"]
        entry["count"] += 1

    candidates = []
    for data in by_ticker.values():
        count = max(data["count"], 1)
        # Normalise scores to 0-1
        superstar_conv = min(data["superstar_hits"] / 3.0, 1.0)
        policy_tw      = min(data["policy_hits"]    / 2.0, 1.0)
        # If both zero but appeared multiple times, still give partial credit
        if superstar_conv == 0.0 and policy_tw == 0.0:
            superstar_conv = min(count / 5.0, 0.5)

        candidates.append(
            StockCandidate(
                ticker=data["ticker"],
                company=data["company"] or data["ticker"],
                sector=data["sector"],
                market=data["market"],
                superstar_conviction=round(superstar_conv, 3),
                policy_tailwind=round(policy_tw, 3),
                investors_backing=list(set(data["investors"])),
                policy_catalysts=list(set(data["policies"])),
            )
        )

    # Sort by initial pre-deep-dive score
    candidates.sort(
        key=lambda c: c.superstar_conviction + c.policy_tailwind,
        reverse=True,
    )
    return candidates


# ── lightweight per-stock deep-dive ───────────────────────────────────────────

def _deep_dive(candidate: StockCandidate) -> StockCandidate:
    """
    Run financial + valuation + news agents on a single candidate.
    Fills in financial_score, valuation_score, news_sentiment, rationale.
    """
    from agents.specialist import financial_agent, news_agent, valuation_agent

    ticker  = candidate.ticker
    company = candidate.company
    market  = candidate.market
    sector  = candidate.sector

    dd_claims: list[SourcedClaim] = []
    errors: list[str] = []

    for name, fn in [
        ("financial",  lambda: financial_agent.run(company, ticker, market)),
        ("valuation",  lambda: valuation_agent.run(company, ticker, market)),
        ("news",       lambda: news_agent.run(company, market)),
    ]:
        try:
            result, _ = fn()
            if isinstance(result, list):
                dd_claims.extend(result)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log.warning("deep_dive_step_failed", ticker=ticker, step=name, error=str(exc))

    candidate.deep_dive_claims = dd_claims

    # Ask LLM to score the candidate from collected claims — only when real data exists
    snippets = [
        str(c.value)[:300] for c in dd_claims
        if isinstance(c.value, dict)
    ][:12]
    combined = "\n\n".join(snippets)

    if not combined.strip():
        # No real data gathered — use initial conviction scores directly
        log.warning("deep_dive_no_data_using_defaults", ticker=ticker)
        candidate.financial_score  = 0.5
        candidate.valuation_score  = 0.5
        candidate.news_sentiment   = 0.5
        candidate.rationale        = "Insufficient data for detailed scoring; ranked by superstar conviction and policy tailwind."
    else:
        prompt = [
            {
                "role": "system",
                "content": (
                    f"Company: {company} ({ticker}), Market: {market}, Sector: {sector}. "
                    "You are a financial analyst. Based on the data snippets, score this stock. "
                    "Return JSON ONLY: "
                    '{"financial_score": float 0-1, '
                    '"valuation_score": float 0-1 (1=undervalued), '
                    '"news_sentiment": float 0-1 (1=very positive), '
                    '"rationale": str (2-3 sentences summarising investment case)}. '
                    "financial_score: 1 means excellent fundamentals; "
                    "valuation_score: 1 means highly undervalued; "
                    "news_sentiment: 1 means very positive recent news."
                ),
            },
            {"role": "user", "content": combined[:3000]},
        ]
        try:
            raw = get_llm_json_response(prompt)
            parsed = json.loads(raw)
            candidate.financial_score  = float(parsed.get("financial_score", 0.5))
            candidate.valuation_score  = float(parsed.get("valuation_score", 0.5))
            candidate.news_sentiment   = float(parsed.get("news_sentiment", 0.5))
            candidate.rationale        = str(parsed.get("rationale", ""))
        except Exception as exc:
            log.warning("deep_dive_llm_failed", ticker=ticker, error=str(exc))
            candidate.financial_score  = 0.5
            candidate.valuation_score  = 0.5
            candidate.news_sentiment   = 0.5
            candidate.rationale        = "Data insufficient for scoring."

    # Composite score: weighted average
    candidate.composite_score = round(
        0.25 * candidate.superstar_conviction
        + 0.20 * candidate.policy_tailwind
        + 0.25 * candidate.financial_score
        + 0.20 * candidate.valuation_score
        + 0.10 * candidate.news_sentiment,
        4,
    )
    return candidate


# ── main entry point ───────────────────────────────────────────────────────────

def run_discovery(market: str) -> DiscoveryResult:
    """
    Run the full discovery pipeline for a given market.

    Parameters
    ----------
    market : str
        "US" or "INDIA"

    Returns
    -------
    DiscoveryResult
    """
    t0 = time.perf_counter()
    market_upper = market.strip().upper()
    errors: list[str] = []

    log.info("discovery_start", market=market_upper)

    # ── Step 1: run SuperstarAgent + GovtSchemeAgent in parallel ──────────────
    from agents.specialist import superstar_agent, govt_scheme_agent

    superstar_claims: list[SourcedClaim] = []
    scheme_claims:    list[SourcedClaim] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_superstar = ex.submit(superstar_agent.run, market_upper)
        fut_schemes   = ex.submit(govt_scheme_agent.run, market_upper)

        for fut in as_completed([fut_superstar, fut_schemes]):
            try:
                result_claims, _ = fut.result(timeout=180)
                if fut is fut_superstar:
                    superstar_claims = result_claims
                else:
                    scheme_claims = result_claims
            except FuturesTimeoutError:
                errors.append("parallel_gather: timeout")
                log.warning("discovery_parallel_timeout")
            except Exception as exc:
                errors.append(f"parallel_gather: {exc}")
                log.warning("discovery_parallel_failed", error=str(exc))

    all_claims = superstar_claims + scheme_claims
    log.info("discovery_gathered", total_claims=len(all_claims))

    # ── Step 2: extract and consolidate candidates ────────────────────────────
    raw_candidates = _extract_candidates_from_claims(all_claims, market_upper)
    candidates     = _consolidate_candidates(raw_candidates, market_upper)

    log.info("discovery_candidates", count=len(candidates))

    # Limit to top MAX_CANDIDATES before deep-dive
    candidates_to_analyse = candidates[:MAX_CANDIDATES]

    # ── Step 3: deep-dive on top candidates in parallel ───────────────────────
    analysed: list[StockCandidate] = []
    if candidates_to_analyse:
        with ThreadPoolExecutor(max_workers=DEEP_DIVE_WORKERS) as ex:
            futures = {ex.submit(_deep_dive, c): c for c in candidates_to_analyse}
            for fut in as_completed(futures):
                cand = futures[fut]
                try:
                    analysed.append(fut.result(timeout=120))
                except FuturesTimeoutError:
                    errors.append(f"deep_dive_timeout({cand.ticker})")
                    log.warning("deep_dive_timeout", ticker=cand.ticker)
                    analysed.append(cand)  # include with partial scores
                except Exception as exc:
                    errors.append(f"deep_dive({cand.ticker}): {exc}")
                    log.warning("deep_dive_failed", ticker=cand.ticker, error=str(exc))
                    analysed.append(cand)  # include with partial scores
    else:
        log.warning("discovery_no_candidates", market=market_upper)

    # ── Step 4: final ranking ─────────────────────────────────────────────────
    analysed.sort(key=lambda c: c.composite_score, reverse=True)
    top_picks = analysed[:TOP_PICKS]

    elapsed = time.perf_counter() - t0
    log.info(
        "discovery_done",
        market=market_upper,
        candidates_evaluated=len(analysed),
        top_picks=len(top_picks),
        elapsed=round(elapsed, 2),
    )

    return DiscoveryResult(
        market=market_upper,
        elapsed_seconds=round(elapsed, 2),
        candidates_evaluated=len(analysed),
        top_picks=top_picks,
        all_superstar_claims=superstar_claims,
        all_scheme_claims=scheme_claims,
        errors=errors,
    )
