"""
frontend/components/lime_panel.py
-----------------------------------
Horizontal bar chart for LIME feature importances.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_lime_panel(feature_importances: list[dict], agent_name: str) -> None:
    if not feature_importances:
        st.info(f"No LIME explanation available for agent: {agent_name}")
        return

    names = [fi["feature_name"] for fi in feature_importances]
    values = [fi["importance"] for fi in feature_importances]
    colors = ["seagreen" if v >= 0 else "crimson" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"LIME Feature Importances — {agent_name}",
        xaxis_title="Importance",
        height=max(300, len(names) * 25),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
