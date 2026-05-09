"""
frontend/components/projection_panel.py
-----------------------------------------
Fan charts with Bull / Base / Bear toggle for EPS, Revenue, Price.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_projection_panel(projections: list[dict], title: str, y_label: str) -> None:
    """
    projections: list of {"year": int, "bull": float, "base": float, "bear": float}
    """
    if not projections:
        st.info(f"No projection data available: {title}")
        return

    scenario = st.radio(
        f"{title} scenario",
        ["All", "Bull", "Base", "Bear"],
        horizontal=True,
        key=f"scenario_{title}",
    )

    years = [str(p["year"]) for p in projections]
    bulls = [p["bull"] for p in projections]
    bases = [p["base"] for p in projections]
    bears = [p["bear"] for p in projections]

    fig = go.Figure()

    if scenario in ("All", "Bull"):
        fig.add_trace(go.Scatter(x=years, y=bulls, name="Bull", line=dict(color="seagreen", dash="dash")))
    if scenario in ("All", "Base"):
        fig.add_trace(go.Scatter(x=years, y=bases, name="Base", line=dict(color="steelblue")))
    if scenario in ("All", "Bear"):
        fig.add_trace(go.Scatter(x=years, y=bears, name="Bear", line=dict(color="crimson", dash="dot")))

    if scenario == "All":
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=bulls + bears[::-1],
            fill="toself",
            fillcolor="rgba(70,130,180,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="Range",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=y_label,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
