"""
retrieval/vector_store.py
--------------------------
Weaviate wrapper with schema-per-domain and TTL-based cache.

Domains (Weaviate class names):
  - FinancialData   (TTL 24h)
  - NewsArticles    (TTL  1h)
  - LegalRecords    (TTL 24h)
  - ManagementData  (TTL 24h)
  - ProductData     (TTL 24h)
  - TenderData      (TTL 24h)
  - OwnershipData   (TTL 24h)

Usage
-----
store = VectorStore()
store.upsert("FinancialData", doc_id, {"ticker": "AAPL", "text": "..."}, vector=[...])
results = store.search("FinancialData", query_vector, top_k=5, filters={"ticker": "AAPL"})
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import structlog
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from config.settings import settings

log = structlog.get_logger(__name__)

# Domain → TTL in seconds
_DOMAIN_TTL: dict[str, int] = {
    "FinancialData": settings.cache_ttl_financials,
    "NewsArticles": settings.cache_ttl_news,
    "LegalRecords": settings.cache_ttl_financials,
    "ManagementData": settings.cache_ttl_financials,
    "ProductData": settings.cache_ttl_financials,
    "TenderData": settings.cache_ttl_financials,
    "OwnershipData": settings.cache_ttl_financials,
    "ValuationData": settings.cache_ttl_financials,
    "CultureData": settings.cache_ttl_financials,
    "InnovationData": settings.cache_ttl_financials,
    "PeerData": settings.cache_ttl_financials,
    "RatingsData": settings.cache_ttl_financials,
}

_VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


class VectorStore:
    """Thin wrapper around Weaviate v4 client."""

    def __init__(self) -> None:
        self._client: weaviate.WeaviateClient | None = None

    def _connect(self) -> weaviate.WeaviateClient:
        if self._client is None or not self._client.is_connected():
            import weaviate.config as _wc
            from urllib.parse import urlparse
            _parsed = urlparse(settings.weaviate_url)
            host = _parsed.hostname or "localhost"
            port = _parsed.port or 8080
            auth = (
                weaviate.auth.AuthApiKey(settings.weaviate_api_key)
                if settings.weaviate_api_key
                else None
            )
            try:
                self._client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    auth_credentials=auth,
                    additional_config=_wc.AdditionalConfig(
                        timeout=_wc.Timeout(init=3, query=5, insert=5)
                    ),
                )
            except Exception as exc:
                raise ConnectionError(f"Weaviate not reachable at {settings.weaviate_url}: {exc}") from exc
            log.info("weaviate_connected", url=settings.weaviate_url)
        return self._client

    def ensure_schema(self) -> None:
        """Create Weaviate collections if they don't exist yet."""
        client = self._connect()
        existing = {c.name for c in client.collections.list_all().values()}

        for domain in _DOMAIN_TTL:
            if domain not in existing:
                try:
                    client.collections.create(
                        name=domain,
                        properties=[
                            Property(name="doc_id", data_type=DataType.TEXT),
                            Property(name="ticker", data_type=DataType.TEXT),
                            Property(name="domain", data_type=DataType.TEXT),
                            Property(name="text", data_type=DataType.TEXT),
                            Property(name="metadata_json", data_type=DataType.TEXT),
                            Property(name="stored_at", data_type=DataType.DATE),
                        ],
                        vectorizer_config=Configure.Vectorizer.none(),
                    )
                    log.info("weaviate_collection_created", domain=domain)
                except Exception as exc:
                    if "already exists" in str(exc).lower():
                        log.debug("weaviate_collection_already_exists", domain=domain)
                    else:
                        raise

    def upsert(
        self,
        domain: str,
        doc_id: str,
        payload: dict[str, Any],
        vector: list[float],
    ) -> None:
        """
        Insert or replace a document.  Uses doc_id as the stable UUID seed
        so re-inserting the same doc_id overwrites the old record.
        """
        import json

        client = self._connect()
        collection = client.collections.get(domain)

        weaviate_id = _stable_uuid(domain, doc_id)
        properties = {
            "doc_id": doc_id,
            "ticker": payload.get("ticker", ""),
            "domain": domain,
            "text": payload.get("text", ""),
            "metadata_json": json.dumps(
                {k: v for k, v in payload.items() if k not in ("text", "ticker")}
            ),
            "stored_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        try:
            collection.data.insert(
                properties=properties,
                vector=vector,
                uuid=weaviate_id,
            )
        except Exception:
            # Object already exists – update it
            collection.data.replace(
                properties=properties,
                vector=vector,
                uuid=weaviate_id,
            )

    def search(
        self,
        domain: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        ttl_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Nearest-neighbour search. Filters stale docs beyond `ttl_seconds`.
        Returns a list of payload dicts (text + metadata).
        """
        import json
        from datetime import timedelta

        client = self._connect()
        collection = client.collections.get(domain)

        weaviate_filter = None
        if filters:
            conditions = [
                Filter.by_property(k).equal(v) for k, v in filters.items()
            ]
            weaviate_filter = conditions[0]
            for cond in conditions[1:]:
                weaviate_filter = weaviate_filter & cond

        if ttl_seconds is None:
            ttl_seconds = _DOMAIN_TTL.get(domain, settings.cache_ttl_financials)

        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(seconds=ttl_seconds)
        ).isoformat()
        ttl_filter = Filter.by_property("stored_at").greater_than(cutoff)
        combined_filter = (
            (weaviate_filter & ttl_filter) if weaviate_filter else ttl_filter
        )

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=combined_filter,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            meta_json = obj.properties.get("metadata_json", "{}")
            extra = json.loads(meta_json) if meta_json else {}
            results.append(
                {
                    "doc_id": obj.properties.get("doc_id", ""),
                    "text": obj.properties.get("text", ""),
                    "ticker": obj.properties.get("ticker", ""),
                    "distance": obj.metadata.distance if obj.metadata else None,
                    **extra,
                }
            )
        return results

    def close(self) -> None:
        if self._client and self._client.is_connected():
            self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton (import and reuse across the app)
# ---------------------------------------------------------------------------
vector_store = VectorStore()


def _stable_uuid(domain: str, doc_id: str) -> str:
    """Generate a deterministic UUID v5-like hex string from domain + doc_id."""
    raw = hashlib.md5(f"{domain}:{doc_id}".encode()).hexdigest()
    # Format as UUID string: 8-4-4-4-12
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"
