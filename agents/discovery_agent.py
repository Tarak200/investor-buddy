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
import yfinance as yf

from config.settings import settings

# get_llm_json_response is a helper that sends a prompt to the LLM and expects a JSON response
from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)

# ── India market-cap tier boundaries (₹ crores) ───────────────────────────────
# Micro Cap  : < ₹1,000 cr
# Small Cap  : ₹1,000 – ₹8,000 cr
# Mid Cap    : ₹8,000 – ₹40,000 cr
# Large Cap  : ₹40,000 – ₹4,00,000 cr
# Mega Cap   : > ₹4,00,000 cr
_INDIA_CAP_RANGES: list[tuple[str, float, float]] = [
    ("Micro Cap",      0.0,        1_000.0),
    ("Small Cap",  1_000.0,        8_000.0),
    ("Mid Cap",    8_000.0,       40_000.0),
    ("Large Cap", 40_000.0,      400_000.0),
    ("Mega Cap",  400_000.0, float("inf")),
]


def _classify_india_market_cap(cr: float) -> str:
    """Return the market-cap tier label for an Indian stock given its market cap in ₹ crores."""
    for tier, low, high in _INDIA_CAP_RANGES:
        if low <= cr < high:
            return tier
    return "Mega Cap"


def _fetch_market_cap_cr(ticker: str) -> float:
    """
    Fetch the numeric market cap in ₹ crores for an Indian stock via yfinance.
    Tries .NS suffix first, then .BO. Returns 0.0 on failure.
    """
    base = ticker.replace(".NS", "").replace(".BO", "")
    for suffix in (".NS", ".BO"):
        try:
            mc = yf.Ticker(base + suffix).fast_info.market_cap
            if mc and mc > 0:
                return float(mc) / 1e7  # INR → ₹ crores
        except Exception:
            pass
    return 0.0


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
    market_cap: str = ""           # e.g. "Mid Cap", "Large Cap" — filled by LLM tools / reclassified
    market_cap_cr: float = 0.0     # numeric market cap in ₹ crores (India only; 0 = unknown)
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
                            "ticker":     item.get("ticker", "").upper().strip(),
                            "company":    item.get("company", ""),
                            "sector":     item.get("sector", "General"),
                            "market_cap": item.get("market_cap", ""),
                            "investor":   item.get("investor", item.get("institution", "")),
                            "policy":     item.get("policy_catalyst",
                                          item.get("benefit_description",
                                          item.get("rationale", ""))),
                            "action":     item.get("action", ""),
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
                "ticker":     ticker,
                "company":    item["company"],
                "sector":     item["sector"],
                "market_cap": item.get("market_cap", ""),
                "market":     market.upper(),
                "investors":  [],
                "policies":   [],
                "superstar_hits": 0,
                "policy_hits":    0,
                "confidence_sum": 0.0,
                "count": 0,
            }
        entry = by_ticker[ticker]
        # Keep best company name
        if item["company"] and not entry["company"]:
            entry["company"] = item["company"]
        # Keep first non-empty market_cap
        if item.get("market_cap") and not entry["market_cap"]:
            entry["market_cap"] = item["market_cap"]
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
                market_cap=data.get("market_cap", ""),
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

def run_discovery(
    market: str,
    *,
    sector: str | None = None,
    market_caps: list[str] | None = None,
) -> DiscoveryResult:
    """
    Run the full discovery pipeline for a given market.

    Parameters
    ----------
    market : str
        "US" or "INDIA"
    sector : str | None
        Optional sector filter (e.g. "IT & Technology").  None means all sectors.
    market_caps : list[str] | None
        Optional list of market-cap tiers to restrict results to.
        Accepted values: "Micro Cap", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap".
        None or empty means all sizes.

    Returns
    -------
    DiscoveryResult
    """
    t0 = time.perf_counter()
    market_upper = market.strip().upper()
    errors: list[str] = []

    log.info("discovery_start", market=market_upper, sector=sector, market_caps=market_caps)

    # ── Step 1: run SuperstarAgent + GovtSchemeAgent in parallel ──────────────
    from agents.specialist import superstar_agent, govt_scheme_agent

    superstar_claims: list[SourcedClaim] = []
    scheme_claims:    list[SourcedClaim] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_superstar = ex.submit(superstar_agent.run, market_upper, sector or "", market_caps)
        fut_schemes   = ex.submit(govt_scheme_agent.run, market_upper, sector or "", market_caps)

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

    # ── For India: fetch real market caps and reclassify tier labels ──────────
    if market_upper == "INDIA" and candidates:
        log.info("discovery_fetching_market_caps", count=len(candidates))
        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as ex:
            mc_futures = {ex.submit(_fetch_market_cap_cr, c.ticker): c for c in candidates}
            for fut in as_completed(mc_futures):
                cand = mc_futures[fut]
                try:
                    mc_cr = fut.result(timeout=10)
                    if mc_cr > 0:
                        cand.market_cap_cr = round(mc_cr, 2)
                        cand.market_cap = _classify_india_market_cap(mc_cr)
                except Exception as exc:
                    log.debug("market_cap_fetch_failed", ticker=cand.ticker, error=str(exc))

    # ── Apply sector & market-cap filters ─────────────────────────────────────
    sector_upper = sector.strip() if sector else None

    # Canonical alias map so "Mid Cap" matches "midcap", "mid cap", "mid-cap" etc.
    _CAP_ALIASES: dict[str, list[str]] = {
        "Micro Cap":  ["micro", "micro cap", "micro-cap", "microcap"],
        "Small Cap":  ["small", "small cap", "small-cap", "smallcap"],
        "Mid Cap":    ["mid", "mid cap", "mid-cap", "midcap"],
        "Large Cap":  ["large", "large cap", "large-cap", "largecap"],
        "Mega Cap":   ["mega", "mega cap", "mega-cap", "megacap"],
    }
    # Exact canonical tier names requested (used for numeric India check)
    requested_tiers: set[str] = set(market_caps) if market_caps else set()
    active_cap_aliases: set[str] = set()
    if market_caps:
        for cap in market_caps:
            active_cap_aliases.update(_CAP_ALIASES.get(cap, [cap.lower()]))

    def _passes_filters(c: StockCandidate) -> bool:
        if sector_upper and c.sector.lower() != sector_upper.lower():
            return False
        if requested_tiers:
            # For India stocks with a fetched numeric market cap, enforce ranges precisely
            if market_upper == "INDIA" and c.market_cap_cr > 0:
                if _classify_india_market_cap(c.market_cap_cr) not in requested_tiers:
                    return False
            elif active_cap_aliases and c.market_cap:
                cap_lower = c.market_cap.lower()
                if not any(alias in cap_lower for alias in active_cap_aliases):
                    return False
        return True

    if sector_upper or active_cap_aliases:
        pre_filter = len(candidates)
        candidates = [c for c in candidates if _passes_filters(c)]
        log.info(
            "discovery_filtered",
            sector=sector_upper,
            market_caps=market_caps,
            before=pre_filter,
            after=len(candidates),
        )

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
