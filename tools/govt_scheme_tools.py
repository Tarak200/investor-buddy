"""
tools/govt_scheme_tools.py
---------------------------
Scans government policies, regulatory notifications, budget schemes, and sector
incentives / penalties that can materially affect equity sectors.

India coverage:
  - Annual Union Budget (PLI schemes, capex, tax changes)
  - SEBI circulars and market regulations
  - RBI monetary policy and sector-specific directions
  - Ministry of Finance / Commerce / Industry notifications
  - National infrastructure and green-energy mandates (e.g., National Green Hydrogen Mission,
    Production Linked Incentive, Make in India, Startup India)

US coverage:
  - Inflation Reduction Act (IRA) — clean energy, EVs, manufacturing
  - CHIPS and Science Act — semiconductors
  - Federal Reserve interest-rate policy impacts
  - SEC rule changes affecting sectors
  - Executive Orders affecting specific industries
  - Congressional bills / reconciliation acts
"""

from __future__ import annotations

import json

import structlog
from langchain_core.tools import tool

from llm.provider import get_llm_json_response
from models.sourced_claim import SourcedClaim
from tools._base import make_claim
from tools.news_tools import _tavily_search

log = structlog.get_logger(__name__)

# ── known standing policies for quick context ─────────────────────────────────

_INDIA_BASE_CONTEXT = (
    "Major ongoing India government schemes as of 2025: "
    "PLI (Production Linked Incentive) for electronics, pharma, auto, solar, textiles, food; "
    "National Green Hydrogen Mission; PM Gati Shakti (infrastructure); "
    "Startup India / DPIIT incentives; SEBI T+0 settlement; RBI rate cycle; "
    "Digital India; National Semiconductor Mission; Defence indigenization (iDEX); "
    "Jal Jeevan Mission (water infra); PMAY housing scheme."
)

_US_BASE_CONTEXT = (
    "Major ongoing US government policies as of 2025: "
    "Inflation Reduction Act – clean energy, EV tax credits, battery manufacturing; "
    "CHIPS and Science Act – semiconductor domestic manufacturing subsidies; "
    "Infrastructure Investment and Jobs Act – roads, bridges, broadband; "
    "Bipartisan Budget Act provisions; SEC AI-related disclosure rules; "
    "Fed rate-cut cycle impact on rate-sensitive sectors; "
    "Export controls on advanced chips (BIS rules); "
    "Tariff landscape (Section 301 China tariffs, Section 232 steel/aluminum); "
    "FTC antitrust enforcement – big tech mergers."
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_scheme_impacts(snippets: list[str], market: str) -> list[dict]:
    """Ask LLM to extract sector incentives / penalties from raw text."""
    if not snippets:
        return []
    base_ctx = _INDIA_BASE_CONTEXT if market.upper() == "INDIA" else _US_BASE_CONTEXT
    combined = "\n\n---\n\n".join(snippets[:8])
    prompt = [
        {
            "role": "system",
            "content": (
                f"Context: {base_ctx}\n\n"
                "You are a policy analysis assistant. "
                "Extract government schemes, policies, or regulatory changes that "
                "INCENTIVIZE (benefit) or PENALIZE (harm) specific sectors. "
                "Return JSON: "
                '{"schemes": [{"name": str, "type": "incentive"|"penalty", '
                '"sectors_affected": [str], "description": str, '
                '"potential_beneficiaries": [str], "magnitude": "low"|"medium"|"high"}]}. '
                "Return empty list if nothing actionable found."
            ),
        },
        {"role": "user", "content": combined[:4000]},
    ]
    try:
        raw = get_llm_json_response(prompt)
        parsed = json.loads(raw)
        return parsed.get("schemes", [])
    except Exception as exc:
        log.warning("scheme_extraction_failed", market=market, error=str(exc))
        return []


# ── India tools ────────────────────────────────────────────────────────────────

@tool
def get_india_govt_schemes(sector: str = "") -> list[SourcedClaim]:
    """
    Fetch latest India government schemes, budget allocations, SEBI/RBI policy changes
    that incentivize or penalize sectors.  Pass sector="" for a broad sweep.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sector_part = f"for {sector} sector" if sector else ""
    queries = [
        f"India government scheme 2025 incentive penalty {sector_part} PLI budget",
        f"SEBI new regulation circular 2025 impact {sector_part}",
        f"RBI policy 2025 sector impact {sector_part}",
        "India budget 2025-26 new scheme sector incentive allocation",
        "India PLI scheme new sectors approved 2025",
    ]
    snippets: list[str] = []
    for q in queries:
        results = _tavily_search(q, max_results=6)
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(content[:600])
                claims.append(
                    make_claim(
                        value={"query": q, "raw": content[:300], "market": "INDIA"},
                        source_url=url,
                        source_name="India Govt Scheme – Web",
                        raw_snippet=content[:500],
                        confidence=0.68,
                    )
                )

    schemes = _extract_scheme_impacts(snippets, market="INDIA")
    if schemes:
        claims.append(
            make_claim(
                value={"india_schemes": schemes, "market": "INDIA"},
                source_url="https://pib.gov.in",
                source_name="India Govt Schemes – Structured",
                raw_snippet=json.dumps(schemes)[:500],
                confidence=0.74,
            )
        )

    log.info("india_schemes_done", sector=sector, claims=len(claims))
    return claims


@tool
def get_india_budget_highlights() -> list[SourcedClaim]:
    """
    Fetch the key sector-level budget highlights and allocations from the latest
    India Union Budget (2025-26).  Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    queries = [
        "India Union Budget 2025-26 sector allocations highlights key changes",
        "Budget 2025 India infrastructure defence agriculture energy allocation",
        "India budget 2025 tax changes LTCG STCG new rules",
        "India finance ministry budget speech 2025 sectors",
    ]
    snippets: list[str] = []
    for q in queries:
        results = _tavily_search(q, max_results=6)
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(content[:600])
                claims.append(
                    make_claim(
                        value={"query": q, "raw": content[:300]},
                        source_url=url,
                        source_name="India Budget 2025-26",
                        raw_snippet=content[:500],
                        confidence=0.72,
                    )
                )

    schemes = _extract_scheme_impacts(snippets, market="INDIA")
    if schemes:
        claims.append(
            make_claim(
                value={"budget_schemes": schemes, "market": "INDIA"},
                source_url="https://www.indiabudget.gov.in",
                source_name="India Budget 2025-26 – Structured",
                raw_snippet=json.dumps(schemes)[:500],
                confidence=0.78,
            )
        )

    log.info("india_budget_done", claims=len(claims))
    return claims


# ── US tools ───────────────────────────────────────────────────────────────────

@tool
def get_us_govt_policies(sector: str = "") -> list[SourcedClaim]:
    """
    Fetch latest US government policies, executive orders, congressional acts and
    regulatory changes that incentivize or penalize sectors.
    Pass sector="" for a broad sweep.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sector_part = f"{sector} sector" if sector else ""
    queries = [
        f"US government policy 2025 incentive penalty {sector_part} IRA CHIPS",
        f"SEC new regulation 2025 impact {sector_part}",
        f"US executive order 2025 {sector_part} industry",
        "US Federal Reserve interest rate 2025 sector impact equities",
        "US tariff 2025 China trade war sector impact companies",
        "Congress bill 2025 subsidy grant new industry incentive",
    ]
    snippets: list[str] = []
    for q in queries:
        results = _tavily_search(q, max_results=6)
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(content[:600])
                claims.append(
                    make_claim(
                        value={"query": q, "raw": content[:300], "market": "US"},
                        source_url=url,
                        source_name="US Govt Policy – Web",
                        raw_snippet=content[:500],
                        confidence=0.68,
                    )
                )

    schemes = _extract_scheme_impacts(snippets, market="US")
    if schemes:
        claims.append(
            make_claim(
                value={"us_schemes": schemes, "market": "US"},
                source_url="https://www.congress.gov",
                source_name="US Govt Policies – Structured",
                raw_snippet=json.dumps(schemes)[:500],
                confidence=0.74,
            )
        )

    log.info("us_policies_done", sector=sector, claims=len(claims))
    return claims


@tool
def get_us_ira_chips_beneficiaries() -> list[SourcedClaim]:
    """
    Identify specific US companies / sectors that are beneficiaries of the
    Inflation Reduction Act (IRA) and CHIPS and Science Act.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    queries = [
        "IRA Inflation Reduction Act beneficiary companies stocks 2025",
        "CHIPS Act semiconductor companies receiving grants 2025",
        "clean energy EV battery companies IRA tax credits 2025",
        "domestic semiconductor fab CHIPS act award companies",
    ]
    snippets: list[str] = []
    for q in queries:
        results = _tavily_search(q, max_results=6)
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(content[:600])
                claims.append(
                    make_claim(
                        value={"query": q, "raw": content[:300]},
                        source_url=url,
                        source_name="IRA/CHIPS Beneficiaries – Web",
                        raw_snippet=content[:500],
                        confidence=0.72,
                    )
                )

    if snippets:
        combined = "\n\n---\n\n".join(snippets[:6])
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst. "
                    "From the text, identify specific companies or sub-sectors that "
                    "are direct beneficiaries of the IRA or CHIPS Act. "
                    "Return JSON: "
                    '{"beneficiaries": [{"ticker": str, "company": str, "sector": str, '
                    '"act": "IRA"|"CHIPS"|"both", "benefit_description": str}]}. '
                    "Return empty list if nothing specific found."
                ),
            },
            {"role": "user", "content": combined[:4000]},
        ]
        try:
            raw = get_llm_json_response(prompt)
            parsed = json.loads(raw)
            beneficiaries = parsed.get("beneficiaries", [])
            if beneficiaries:
                claims.append(
                    make_claim(
                        value={"ira_chips_beneficiaries": beneficiaries, "market": "US"},
                        source_url="https://www.congress.gov",
                        source_name="IRA/CHIPS Beneficiaries – Structured",
                        raw_snippet=json.dumps(beneficiaries)[:500],
                        confidence=0.75,
                    )
                )
        except Exception as exc:
            log.warning("ira_chips_extraction_failed", error=str(exc))
    else:
        # Tavily unavailable — inject known IRA/CHIPS beneficiaries as fallback
        log.warning("ira_chips_no_data_fallback")
        fallback_beneficiaries = [
            {"ticker": "ENPH", "company": "Enphase Energy", "sector": "Clean Energy", "act": "IRA", "benefit_description": "Solar microinverter manufacturing tax credits"},
            {"ticker": "FSLR", "company": "First Solar", "sector": "Clean Energy", "act": "IRA", "benefit_description": "US-made thin-film solar panel production incentives"},
            {"ticker": "INTC", "company": "Intel Corporation", "sector": "Semiconductors", "act": "CHIPS", "benefit_description": "Domestic fab expansion grants and loans"},
            {"ticker": "TSM", "company": "Taiwan Semiconductor (US ADR)", "sector": "Semiconductors", "act": "CHIPS", "benefit_description": "Arizona fab CHIPS Act award recipient"},
            {"ticker": "RIVN", "company": "Rivian Automotive", "sector": "EV", "act": "IRA", "benefit_description": "Commercial EV tax credits and battery manufacturing"},
        ]
        claims.append(
            make_claim(
                value={"ira_chips_beneficiaries": fallback_beneficiaries, "market": "US"},
                source_url="https://www.congress.gov",
                source_name="IRA/CHIPS Beneficiaries – Fallback",
                raw_snippet=json.dumps(fallback_beneficiaries)[:500],
                confidence=0.50,
            )
        )

    log.info("ira_chips_done", claims=len(claims))
    return claims


# ── Cross-market tool ──────────────────────────────────────────────────────────

@tool
def get_policy_driven_stock_ideas(market: str, sector: str = "", market_cap_filter: str = "") -> list[SourcedClaim]:
    """
    High-level scan: combine government scheme analysis with sector screening to
    identify specific stock tickers that could benefit from current policy tailwinds.
    market: "US" | "INDIA"
    sector: optional sector filter (e.g. "Defence"); empty = all sectors.
    market_cap_filter: optional comma-separated market-cap tiers; empty = all sizes.
    Returns List[SourcedClaim].
    """
    claims: list[SourcedClaim] = []
    sector_part = f" {sector}" if sector else ""

    if market.upper() == "INDIA":
        queries = [
            f"India government scheme 2025 beneficiary{sector_part} stocks NSE PLI defence railway",
            f"India budget 2025{sector_part} infrastructure renewable energy stocks to buy",
            f"India government capex spending{sector_part} beneficiary companies 2025",
            f"India EV policy solar energy{sector_part} stocks beneficiary 2025",
        ]
        source_base = "https://pib.gov.in"
    else:
        queries = [
            f"US government policy 2025 beneficiary{sector_part} stocks IRA CHIPS defence",
            f"US infrastructure bill beneficiary{sector_part} stocks NYSE NASDAQ 2025",
            f"US tariff beneficiary domestic{sector_part} manufacturer stocks 2025",
            f"US clean energy EV defence aerospace{sector_part} government contract stocks",
        ]
        source_base = "https://www.congress.gov"

    snippets: list[str] = []
    for q in queries:
        results = _tavily_search(q, max_results=8)
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(content[:600])
                claims.append(
                    make_claim(
                        value={"query": q, "raw": content[:300], "market": market},
                        source_url=url,
                        source_name=f"Policy-Driven Stock Ideas – {market}",
                        raw_snippet=content[:500],
                        confidence=0.65,
                    )
                )

    # Build sector / market-cap instruction fragments for the LLM prompt
    sector_instruction = f" ONLY include stocks in the '{sector}' sector." if sector else ""
    cap_instruction = (
        f" ONLY include stocks with market cap in: {market_cap_filter}."
        if market_cap_filter else ""
    )
    filter_note = sector_instruction + cap_instruction

    if snippets:
        combined = "\n\n---\n\n".join(snippets[:8])
        prompt = [
            {
                "role": "system",
                "content": (
                    f"Market: {market}.{filter_note} "
                    "You are a financial analyst. From the text, extract specific stocks "
                    "or companies that are likely to benefit from current government "
                    "policies, schemes, or regulatory tailwinds. "
                    "Return JSON: "
                    '{"policy_picks": [{"ticker": str, "company": str, "sector": str, '
                    '"market_cap": str, "policy_catalyst": str, "rationale": str}]}. '
                    "Return empty list if nothing actionable."
                ),
            },
            {"role": "user", "content": combined[:4000]},
        ]
        try:
            raw = get_llm_json_response(prompt)
            parsed = json.loads(raw)
            picks = parsed.get("policy_picks", [])
            if picks:
                claims.append(
                    make_claim(
                        value={"policy_picks": picks, "market": market},
                        source_url=source_base,
                        source_name=f"Policy-Driven Picks – {market} Structured",
                        raw_snippet=json.dumps(picks)[:500],
                        confidence=0.72,
                    )
                )
        except Exception as exc:
            log.warning("policy_picks_extraction_failed", market=market, error=str(exc))
    else:
        # Tavily unavailable — inject known policy-driven picks as fallback
        log.warning("policy_picks_no_data_fallback", market=market)
        if market.upper() == "INDIA":
            fallback_picks = [
                {"ticker": "IRFC", "company": "Indian Railway Finance Corp", "sector": "Infrastructure", "policy_catalyst": "Railway capex Rs 2.5 lakh crore", "rationale": "Primary financier of Indian Railways expansion"},
                {"ticker": "NTPC", "company": "NTPC Limited", "sector": "Power", "policy_catalyst": "National Green Hydrogen Mission", "rationale": "Government-backed green energy transition play"},
                {"ticker": "HAL", "company": "Hindustan Aeronautics", "sector": "Defence", "policy_catalyst": "Defence indigenization iDEX", "rationale": "25% defence offset obligation benefits domestic OEMs"},
                {"ticker": "DIXON", "company": "Dixon Technologies", "sector": "Electronics", "policy_catalyst": "PLI scheme for electronics", "rationale": "Leading beneficiary of mobile/electronics PLI"},
                {"ticker": "SOLARIND", "company": "Solar Industries", "sector": "Defence", "policy_catalyst": "Atmanirbhar Bharat defence", "rationale": "Explosives and ammunition for domestic defence"},
            ]
        else:
            fallback_picks = [
                {"ticker": "LMT", "company": "Lockheed Martin", "sector": "Defence", "policy_catalyst": "US defence spending increase", "rationale": "Top defence contractor benefiting from budget increases"},
                {"ticker": "NEE", "company": "NextEra Energy", "sector": "Clean Energy", "policy_catalyst": "IRA clean energy credits", "rationale": "Largest US renewable energy producer"},
                {"ticker": "AMAT", "company": "Applied Materials", "sector": "Semiconductors", "policy_catalyst": "CHIPS Act domestic fab buildout", "rationale": "Semiconductor equipment supplier for new US fabs"},
                {"ticker": "CAT", "company": "Caterpillar", "sector": "Infrastructure", "policy_catalyst": "Infrastructure Investment Act", "rationale": "Construction equipment demand from federal projects"},
                {"ticker": "PLUG", "company": "Plug Power", "sector": "Clean Energy", "policy_catalyst": "IRA hydrogen tax credits", "rationale": "Green hydrogen electrolyser beneficiary"},
            ]
        claims.append(
            make_claim(
                value={"policy_picks": fallback_picks, "market": market},
                source_url=source_base,
                source_name=f"Policy-Driven Picks – {market} Fallback",
                raw_snippet=json.dumps(fallback_picks)[:500],
                confidence=0.50,
            )
        )

    log.info("policy_driven_ideas_done", market=market, claims=len(claims))
    return claims
