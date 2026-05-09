"""
frontend/components/charts.py
------------------------------
Plotly chart renderer component for Streamlit.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_chart(chart_data: dict[str, Any] | None, title: str = "") -> None:
    """Render a Plotly figure dict in Streamlit."""
    if not chart_data:
        st.info(f"No chart data available for: {title}")
        return
    try:
        fig = go.Figure(chart_data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart render error ({title}): {exc}")
