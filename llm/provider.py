"""
llm/provider.py
---------------
LiteLLM wrapper with primary (Groq) + fallback (OpenRouter) strategy.
Every agent in the pipeline calls `get_llm_response()` through this module.
"""

from __future__ import annotations

import os
from typing import Any

import litellm
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


_FALLBACK_MODELS: list[str] = [
    m
    for m in [settings.llm_fallback_model_1, settings.llm_fallback_model_2]
    if m
]


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
    except Exception as exc:
        log.error("llm_call_failed", error=str(exc), primary_model=primary)
        raise


def get_llm_json_response(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Convenience wrapper that requests a JSON object response.
    The caller is responsible for parsing the returned string with json.loads().
    """
    return get_llm_response(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
