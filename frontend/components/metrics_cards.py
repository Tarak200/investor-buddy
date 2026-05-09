"""
frontend/components/metrics_cards.py
--------------------------------------
KPI metric card component for Streamlit.
"""

from __future__ import annotations

import streamlit as st


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def render_kpi_row(metrics: list[dict]) -> None:
    """
    Render a row of KPI cards.
    `metrics` = [{"label": str, "value": str, "delta": str | None}, ...]
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric(m["label"], m["value"], m.get("delta"))
