"""
frontend/components/disputed_banner.py
----------------------------------------
Collapsible ⚠️ banner listing disputed claims (hallucination_score > 0.5).
"""

from __future__ import annotations

import streamlit as st


def render_disputed_banner(disputed_claims: list[dict]) -> None:
    if not disputed_claims:
        return
    with st.expander(f"⚠️ {len(disputed_claims)} Disputed / Unverified Claims", expanded=False):
        for i, claim in enumerate(disputed_claims, 1):
            st.markdown(
                f"**{i}.** `{claim.get('source_name', 'Unknown')}` — "
                f"Score: {claim.get('hallucination_score', '?'):.2f} | "
                f"Value: *{str(claim.get('value', ''))[:120]}*"
            )
