"""
frontend/components/charts.py
------------------------------
Plotly chart renderer component for Streamlit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _extract_trace_rows(chart_data: dict) -> list[dict]:
    """Pull x/y (or labels/values) from every trace into a list of row dicts."""
    rows: list[dict] = []
    for trace in chart_data.get("data", []):
        name = trace.get("name", "")
        xs = trace.get("x") or trace.get("labels") or []
        ys = trace.get("y") or trace.get("values") or []
        for x, y in zip(xs, ys):
            rows.append({"Category": x, name or "Value": y})
    return rows


def render_chart(chart_data: dict[str, Any] | None, title: str = "") -> None:
    """Render a Plotly figure dict in Streamlit.

    - If chart_data is empty/None: show an info message directing to the Overview tab.
    - If chart_data is valid: render the chart, then offer a collapsible data table.
    - If rendering itself fails: show the raw data table directly.
    """
    if not chart_data:
        st.info(
            f"No chart data for **{title}** — the underlying data is available "
            "as tables in the **Overview** tab."
        )
        return

    try:
        fig = go.Figure(chart_data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart render error ({title}): {exc}")

    # Always offer raw data table in a collapsible expander
    rows = _extract_trace_rows(chart_data)
    if rows:
        with st.expander(f"📋 View data — {title}", expanded=False):
            try:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.json(rows)
