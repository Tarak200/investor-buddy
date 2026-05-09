"""
retrieval/claim_cache.py
-------------------------
Thin caching layer between specialist agents and Weaviate.

Each agent call that produces SourcedClaims can:
  1. Check the cache first (get_cached_claims) — returns claims on a fresh hit.
  2. Store results after a live fetch (cache_claims) — fire-and-forget, never raises.

If Weaviate is not running, both functions log a warning and return gracefully
so the rest of the pipeline is never blocked.

Usage in a specialist agent
---------------------------
    from retrieval.claim_cache import cache_claims, get_cached_claims

    cached = get_cached_claims(key=ticker, domain="FinancialData",
                               query_hint=f"{company} {ticker} financial statements")
    if cached is not None:
        return cached, time.perf_counter() - t0   # fast cache hit

    # ... live fetch logic ...

    cache_claims(key=ticker, domain="FinancialData", claims=claims,
                 query_hint=f"{company} {ticker} financial statements")
    return claims, elapsed
"""

from __future__ import annotations

import json

import structlog

from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)


def get_cached_claims(
    key: str,
    domain: str,
    query_hint: str,
) -> list[SourcedClaim] | None:
    """
    Return cached claims for (key, domain) if Weaviate holds fresh data within TTL.

    Parameters
    ----------
    key        : ticker symbol or company name used as the Weaviate ticker field.
    domain     : Weaviate collection name (e.g. "FinancialData", "NewsArticles").
    query_hint : A short descriptive text used to build the query embedding.

    Returns
    -------
    list[SourcedClaim] on a cache hit, None on miss or any error.
    """
    try:
        from retrieval.embeddings import embed_single
        from retrieval.vector_store import vector_store

        vector_store.ensure_schema()
        query_vector = embed_single(query_hint)
        results = vector_store.search(
            domain, query_vector, top_k=1, filters={"ticker": key}
        )
        if not results:
            return None

        raw = results[0].get("claims_json")
        if not raw:
            return None

        claim_dicts = json.loads(raw)
        claims = [SourcedClaim.model_validate(d) for d in claim_dicts]
        log.info("cache_hit", domain=domain, key=key, claims=len(claims))
        return claims

    except Exception as exc:
        log.warning("cache_read_failed", domain=domain, key=key, error=str(exc))
        return None


def cache_claims(
    key: str,
    domain: str,
    claims: list[SourcedClaim],
    query_hint: str,
) -> None:
    """
    Store agent claims in Weaviate. Fire-and-forget — errors are logged, never raised.

    Parameters
    ----------
    key        : ticker symbol or company name (stored in the ``ticker`` property).
    domain     : Weaviate collection name (e.g. "FinancialData", "NewsArticles").
    claims     : The list of SourcedClaim objects to persist.
    query_hint : Short descriptive text used to build the storage embedding.
    """
    if not claims:
        return

    try:
        from retrieval.embeddings import embed_single
        from retrieval.vector_store import vector_store

        vector_store.ensure_schema()

        # Build a text blob for vector similarity (capped at 2 000 chars)
        combined_text = " ".join(
            c.raw_snippet for c in claims if c.raw_snippet
        )[:2000] or query_hint

        vector = embed_single(combined_text)

        # Serialize all claims; model_dump_json handles UUID/datetime correctly
        claims_data = [json.loads(c.model_dump_json()) for c in claims]

        vector_store.upsert(
            domain=domain,
            doc_id=f"{key}:{domain}",
            payload={
                "ticker": key,
                "text": combined_text,
                "claims_json": json.dumps(claims_data),
            },
            vector=vector,
        )
        log.info("cache_write", domain=domain, key=key, claims=len(claims))

    except Exception as exc:
        log.warning("cache_write_failed", domain=domain, key=key, error=str(exc))
