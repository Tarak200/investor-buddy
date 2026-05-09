"""
evaluation/latency_monitor.py
-------------------------------
Per-agent latency tracking with context manager and decorator interfaces.
Integrates with LangSmith when LANGCHAIN_API_KEY is set.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Callable

import structlog

log = structlog.get_logger(__name__)


class LatencyRecord:
    __slots__ = ("agent_name", "elapsed_s", "tokens_in", "tokens_out", "cost_usd")

    def __init__(self, agent_name: str, elapsed_s: float, tokens_in: int = 0, tokens_out: int = 0):
        self.agent_name = agent_name
        self.elapsed_s = elapsed_s
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        # Rough cost estimate for llama-3.3-70b on Groq (free tier)
        self.cost_usd = 0.0  # free tier has no cost


_records: list[LatencyRecord] = []


@contextmanager
def track(agent_name: str):
    """Context manager to time an agent block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        record = LatencyRecord(agent_name=agent_name, elapsed_s=round(elapsed, 3))
        _records.append(record)
        log.info("latency_record", agent=agent_name, elapsed_s=record.elapsed_s)
        _push_to_langsmith(record)


def timed(agent_name: str):
    """Decorator to time a function call."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with track(agent_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_summary() -> dict[str, dict]:
    """
    Return per-agent p50/p95 latency statistics.
    """
    from collections import defaultdict
    import numpy as np

    agent_times: dict[str, list[float]] = defaultdict(list)
    for r in _records:
        agent_times[r.agent_name].append(r.elapsed_s)

    summary: dict[str, dict] = {}
    for agent, times in agent_times.items():
        arr = np.array(times)
        summary[agent] = {
            "count": len(arr),
            "p50_s": round(float(np.percentile(arr, 50)), 3),
            "p95_s": round(float(np.percentile(arr, 95)), 3),
            "mean_s": round(float(arr.mean()), 3),
            "max_s": round(float(arr.max()), 3),
        }
    return summary


def reset() -> None:
    """Clear all records (useful between test runs)."""
    _records.clear()


def _push_to_langsmith(record: LatencyRecord) -> None:
    """Push latency record to LangSmith if API key is configured."""
    try:
        from langsmith import Client
        from config.settings import settings
        if not settings.langchain_api_key:
            return
        client = Client(api_key=settings.langchain_api_key)
        client.create_run(
            name=record.agent_name,
            run_type="chain",
            inputs={"agent": record.agent_name},
            outputs={"elapsed_s": record.elapsed_s},
        )
    except Exception:
        pass  # LangSmith push is non-critical
