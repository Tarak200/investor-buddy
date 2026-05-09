"""
frontend/components/department_heatmap.py
-------------------------------------------
Plotly annotated heatmap for sentiment by department.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_department_heatmap(sentiment_by_dept: dict[str, float], company: str) -> None:
    if not sentiment_by_dept:
        st.info("No department sentiment data available.")
        return

    depts = list(sentiment_by_dept.keys())
    scores = [[sentiment_by_dept[d] for d in depts]]
    annotations = [
        dict(x=d, y=0, text=f"{sentiment_by_dept[d]:.1f}", showarrow=False, font=dict(color="white"))
        for d in depts
    ]

    fig = go.Figure(go.Heatmap(
        z=scores,
        x=depts,
        y=[company],
        colorscale="RdYlGn",
        zmin=0,
        zmax=10,
    ))
    fig.update_layout(
        title=f"{company} — Sentiment by Department",
        annotations=annotations,
        height=180,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
