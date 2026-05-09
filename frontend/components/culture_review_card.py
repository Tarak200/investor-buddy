"""
frontend/components/culture_review_card.py
--------------------------------------------
Review excerpt card with platform icon.
"""

from __future__ import annotations

import streamlit as st

_PLATFORM_ICONS = {
    "glassdoor": "🟢",
    "ambitionbox": "🟡",
    "reddit": "🟠",
    "twitter": "🐦",
    "linkedin": "💼",
}


def render_review_card(text: str, platform: str, rating: float | None = None) -> None:
    icon = _PLATFORM_ICONS.get(platform.lower(), "💬")
    rating_str = f" — ⭐ {rating:.1f}" if rating is not None else ""
    st.markdown(
        f"""
        <div style="border:1px solid #333; border-radius:8px; padding:10px; margin:4px 0;">
        {icon} <strong>{platform.title()}</strong>{rating_str}<br/>
        <em>{text[:300]}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
