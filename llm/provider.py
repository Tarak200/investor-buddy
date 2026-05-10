"""
llm/provider.py
---------------
LiteLLM wrapper with primary (Groq) + fallback (OpenRouter) strategy.
Every agent in the pipeline calls `get_llm_response()` through this module.
"""

from __future__ import annotations

import os
import re
import random
import time
from typing import Any

import litellm
from litellm.exceptions import RateLimitError
import structlog

from config.settings import settings

log = structlog.get_logger(__name__)

# Silence LiteLLM's verbose default logging; our structlog handles it.
litellm.set_verbose = False

# Set API keys so LiteLLM can pick them up automatically.
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
if settings.openrouter_api_key:
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
if settings.google_api_key:
    os.environ.setdefault("GEMINI_API_KEY", settings.google_api_key)


_FALLBACK_MODELS: list[str] = [
    m
    for m in [
        settings.llm_fallback_model_1,
        settings.llm_fallback_model_2,
        settings.llm_fallback_model_3,
    ]
    if m
]

# Retry configuration for rate-limit errors
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 8.0  # seconds; doubles each attempt (8, 16, 32, 60)

# Global semaphore: at most 1 concurrent LLM call to avoid hammering rate limits
import threading
_LLM_SEMAPHORE = threading.Semaphore(1)

# Token-rate throttle: enforce minimum gap between calls to respect TPM limits.
# groq/llama-3.1-8b-instant is 6000 TPM; with ~600 tokens/call = 10 calls/min max.
# Enforce 12s minimum between releases so we stay safely under 6000 TPM.
_MIN_CALL_INTERVAL = 12.0  # seconds
_last_call_time: float = 0.0
_throttle_lock = threading.Lock()


def _rate_limit_delay(attempt: int, exc: Exception | None = None) -> float:
    """Parse retry-after from Groq/OpenRouter error message, or use exponential backoff."""
    if exc is not None:
        msg = str(exc)
        # Groq format: "Please try again in 9m5.183999999s."
        m = re.search(r'try again in (?:(\d+)m)?([\d.]+)s', msg)
        if m:
            mins = int(m.group(1) or 0)
            secs = float(m.group(2) or 0)
            parsed = mins * 60 + secs + 2.0  # +2s buffer
            log.warning("llm_retry_after_parsed", retry_after_s=round(parsed, 1))
            return min(parsed, 120.0)  # cap at 2 min
    return min(_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(-1, 1), 90.0)


def get_llm_response(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    Call LLM via LiteLLM with automatic fallback.

    Parameters
    ----------
    messages:
        OpenAI-style message list, e.g.
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    model:
        Override the default primary model.
    temperature:
        Sampling temperature (default 0.1 for analytical tasks).
    max_tokens:
        Maximum tokens to generate.
    response_format:
        Optional dict, e.g. {"type": "json_object"} for structured outputs.

    Returns
    -------
    str
        The assistant's reply text.
    """
    primary = model or settings.llm_primary_model

    kwargs: dict[str, Any] = {
        "model": primary,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _FALLBACK_MODELS:
        kwargs["fallbacks"] = _FALLBACK_MODELS
    if response_format:
        kwargs["response_format"] = response_format

    last_exc: Exception | None = None
    with _LLM_SEMAPHORE:
        # Token-rate throttle: enforce minimum interval between LLM calls
        global _last_call_time
        with _throttle_lock:
            now = time.monotonic()
            gap = now - _last_call_time
            if gap < _MIN_CALL_INTERVAL:
                time.sleep(_MIN_CALL_INTERVAL - gap)
            _last_call_time = time.monotonic()

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = litellm.completion(**kwargs)
                content: str = response.choices[0].message.content or ""
                log.debug(
                    "llm_response_ok",
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                return content
            except RateLimitError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    break
                delay = _rate_limit_delay(attempt, exc)
                log.warning(
                    "llm_rate_limit_retry",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    delay_s=round(delay, 1),
                    primary_model=primary,
                )
                time.sleep(delay)
            except Exception as exc:
                log.warning("llm_call_failed", error=str(exc)[:300], primary_model=primary)
                raise

    log.warning("llm_all_retries_exhausted", error=str(last_exc), primary_model=primary)
    raise last_exc  # type: ignore[misc]


def get_llm_json_response(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Convenience wrapper that requests a JSON object response.
    The caller is responsible for parsing the returned string with json.loads().
    Caps at 2048 tokens to avoid groq json_validate_failed on truncated generation.
    """
    # Truncate the user message to avoid exceeding model context / hitting TPM limits
    capped_messages = []
    for m in messages:
        if m.get("role") == "user" and len(m.get("content", "")) > 3000:
            capped_messages.append({**m, "content": m["content"][:3000]})
        else:
            capped_messages.append(m)
    return get_llm_response(
        capped_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
