"""
retrieval/embeddings.py
-----------------------
Embedding generation using sentence-transformers (local, no API key needed).
Falls back to a simple hash-based dummy if the library is unavailable during tests.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import structlog

log = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the sentence transformer model (cached after first call)."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(_MODEL_NAME)
        log.info("embedding_model_loaded", model=_MODEL_NAME)
        return model
    except ImportError:
        log.warning(
            "sentence_transformers_not_installed",
            fallback="hash-based dummy embeddings",
        )
        return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts.  Returns a list of float vectors (384-dim for MiniLM).
    If sentence-transformers is unavailable, returns deterministic dummy vectors.
    """
    if not texts:
        return []

    model = _get_model()
    if model is not None:
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    # Fallback: deterministic pseudo-embeddings for testing / CI
    result = []
    for text in texts:
        seed = sum(ord(c) for c in text[:100]) % (2**32)
        rng = np.random.default_rng(seed)
        result.append(rng.uniform(-1, 1, size=384).tolist())
    return result


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for a single text."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []
