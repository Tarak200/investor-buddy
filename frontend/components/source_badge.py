"""
frontend/components/source_badge.py
-------------------------------------
Renders a [Source Name ↗] chip with tooltip showing fetched_at date.
"""

from __future__ import annotations

import streamlit as st


def render_source_badge(source_name: str, source_url: str, fetched_at: str | None = None) -> None:
    tooltip = f"Fetched: {fetched_at}" if fetched_at else ""
    link_text = f"[{source_name} ]({source_url})"
    if tooltip:
        st.caption(f"{link_text} *{tooltip}*")
    else:
        st.caption(link_text)


def render_source_list(footnotes: list[dict]) -> None:
    """Render a numbered source list from footnote dicts."""
    for fn in footnotes:
        idx = fn.get("index", "")
        name = fn.get("source_name", "")
        url = fn.get("source_url", "#")
        fetched = fn.get("fetched_at", "")
        st.markdown(f"**[{idx}]** [{name}]({url}) — *{fetched}*")
