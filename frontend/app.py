"""
frontend/app.py
----------------
Streamlit application — Financial Research Copilot.

Tabs (15):
  Overview | Projections | Valuation & Technical | Financials | Ownership |
  Peers | Ratings | News & Sentiment | Legal | Orders & Tenders |
  Products | Management | Culture | Innovation & Global | Sources

Run with:
  streamlit run frontend/app.py
"""

from __future__ import annotations

import time

import requests
import streamlit as st

from frontend.components.charts import render_chart
from frontend.components.culture_review_card import render_review_card
from frontend.components.department_heatmap import render_department_heatmap
from frontend.components.disputed_banner import render_disputed_banner
from frontend.components.lime_panel import render_lime_panel
from frontend.components.metrics_cards import render_kpi_row
from frontend.components.projection_panel import render_projection_panel
from frontend.components.source_badge import render_source_list

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Financial Research Copilot",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Research Copilot")
st.caption("Multi-agent deep-dive analysis for US & Indian equities")

# ── Sidebar — Input Form ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Analyse a Company")
    company = st.text_input("Company Name", placeholder="e.g. Reliance Industries")
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. RELIANCE or AAPL")
    market = st.selectbox("Market", ["INDIA", "US"])
    sector = st.text_input("Sector", placeholder="e.g. Energy, Technology")
    horizon = st.radio(
        "Time Horizon",
        [1, 2, 3, 5, 7, 10],
        format_func=lambda x: f"{x}Y",
        horizontal=True,
    )
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

# ── Trigger Analysis ──────────────────────────────────────────────────────────
if run_btn:
    if not company or not ticker:
        st.error("Please enter both Company Name and Ticker Symbol.")
        st.stop()

    with st.spinner("Queuing analysis job…"):
        try:
            resp = requests.post(
                f"{API_BASE}/analyze",
                json={
                    "company": company,
                    "ticker": ticker,
                    "market": market,
                    "sector": sector or "General",
                    "time_horizon_years": horizon,
                },
                timeout=10,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            st.session_state["job_id"] = job_id
            st.session_state["report"] = None
            st.success(f"Job queued: `{job_id}`")
        except Exception as exc:
            st.error(f"Failed to start analysis: {exc}")
            st.stop()

# ── Poll Status ───────────────────────────────────────────────────────────────
job_id: str | None = st.session_state.get("job_id")
report: dict | None = st.session_state.get("report")

if job_id and not report:
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    agent_order = [
        "financial", "news", "legal", "order_book", "product",
        "management", "ownership", "peer", "culture", "innovation",
        "valuation", "ratings",
    ]
    max_polls = 120  # 2 minutes max
    polls = 0
    while polls < max_polls:
        polls += 1
        try:
            s_resp = requests.get(f"{API_BASE}/status/{job_id}", timeout=5)
            s_resp.raise_for_status()
            s_data = s_resp.json()
        except Exception:
            time.sleep(2)
            continue

        done_agents = sum(1 for ag in s_data.get("agent_statuses", []) if ag["status"] == "done")
        progress = done_agents / max(len(agent_order), 1)
        progress_bar.progress(min(progress, 0.95))
        status_placeholder.info(f"Step: **{s_data.get('current_step', '…')}** — {done_agents}/12 agents done")

        if s_data.get("completed"):
            progress_bar.progress(1.0)
            # Fetch report
            try:
                r_resp = requests.get(f"{API_BASE}/report/{job_id}", timeout=30)
                r_resp.raise_for_status()
                st.session_state["report"] = r_resp.json()
                report = st.session_state["report"]
                status_placeholder.success("Analysis complete!")
            except Exception as exc:
                st.error(f"Failed to fetch report: {exc}")
            break

        time.sleep(3)
    else:
        st.warning("Analysis is taking longer than expected. Refresh to check.")

# ── Render Report ─────────────────────────────────────────────────────────────
if report:
    company_name = report.get("company", "")
    stance = report.get("overall_stance", "NEUTRAL")
    confidence = report.get("overall_confidence", 0.0)
    horizon_out = report.get("time_horizon_years", 1)
    charts = report.get("chart_data") or {}
    forecast = report.get("forecast") or {}
    verification = report.get("verification") or {}

    stance_color = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(stance, "⚪")
    st.subheader(f"{company_name}  {stance_color} {stance}  ({horizon_out}Y Horizon)")

    # KPI Row
    render_kpi_row([
        {"label": "Overall Stance", "value": stance},
        {"label": "Confidence", "value": f"{confidence * 100:.0f}%"},
        {"label": "Total Claims", "value": str(verification.get("total_claims", 0))},
        {"label": "Hallucinated", "value": str(verification.get("hallucinated_claims", 0))},
        {"label": "Verified", "value": str(verification.get("verified_claims", 0))},
    ])

    # 15 Tabs
    tabs = st.tabs([
        "Overview", "Projections", "Valuation & Technical", "Financials",
        "Ownership", "Peers", "Ratings", "News & Sentiment",
        "Legal", "Orders & Tenders", "Products", "Management",
        "Culture", "Innovation & Global", "Sources",
    ])

    # ── Tab 0: Overview ─────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(report.get("markdown_report", "_Report not available._"))

    # ── Tab 1: Projections ──────────────────────────────────────────────────
    with tabs[1]:
        eps_proj = forecast.get("eps_projections", [])
        rev_proj = forecast.get("revenue_projections", [])
        price_proj = forecast.get("price_projections", [])

        render_projection_panel(eps_proj, "EPS Projections", "EPS")
        render_projection_panel(rev_proj, "Revenue Projections", "Revenue (Cr / USD M)")
        render_projection_panel(price_proj, "Price Projections", "Price")

        st.markdown("**Investment Thesis**")
        st.info(forecast.get("investment_thesis", "N/A"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Risks**")
            for r in forecast.get("key_risks", []):
                st.markdown(f"- {r}")
        with col2:
            st.markdown("**Key Catalysts**")
            for c in forecast.get("key_catalysts", []):
                st.markdown(f"- {c}")

        forecast_charts = charts.get("forecast") or {}
        if isinstance(forecast_charts, dict):
            for chart_key in ("eps_fan", "revenue_fan", "price_area", "market_share_bars", "weight_donut"):
                if forecast_charts.get(chart_key):
                    render_chart(forecast_charts[chart_key], chart_key)

    # ── Tab 2: Valuation & Technical ────────────────────────────────────────
    with tabs[2]:
        render_chart(charts.get("valuation"), "Valuation Comparison")
        st.divider()
        st.subheader("Technical Indicators")
        render_chart(charts.get("technicals"), "Technicals")

    # ── Tab 3: Financials ───────────────────────────────────────────────────
    with tabs[3]:
        render_chart(charts.get("financial"), "Revenue & PAT")

    # ── Tab 4: Ownership ────────────────────────────────────────────────────
    with tabs[4]:
        render_chart(charts.get("ownership"), "Ownership Structure")

    # ── Tab 5: Peers ────────────────────────────────────────────────────────
    with tabs[5]:
        render_chart(charts.get("peers"), "Peer Valuation")
        render_chart(charts.get("market_share"), "Market Share")

    # ── Tab 6: Ratings ──────────────────────────────────────────────────────
    with tabs[6]:
        render_chart(charts.get("ratings"), "Broker Consensus")

    # ── Tab 7: News & Sentiment ─────────────────────────────────────────────
    with tabs[7]:
        render_chart(charts.get("sentiment"), "News Sentiment")

    # ── Tab 8: Legal ────────────────────────────────────────────────────────
    with tabs[8]:
        render_chart(charts.get("risk"), "Risk Radar")

    # ── Tab 9: Orders & Tenders ─────────────────────────────────────────────
    with tabs[9]:
        render_chart(charts.get("orders"), "Order Pipeline")

    # ── Tab 10: Products ────────────────────────────────────────────────────
    with tabs[10]:
        st.info("Product analysis data in the Overview tab.")

    # ── Tab 11: Management ──────────────────────────────────────────────────
    with tabs[11]:
        render_chart(charts.get("management"), "Management Radar")

    # ── Tab 12: Culture ─────────────────────────────────────────────────────
    with tabs[12]:
        render_chart(charts.get("culture"), "Culture Dimensions")

    # ── Tab 13: Innovation & Global ─────────────────────────────────────────
    with tabs[13]:
        render_chart(charts.get("innovation"), "R&D Intensity")

    # ── Tab 14: Sources ─────────────────────────────────────────────────────
    with tabs[14]:
        footnotes = report.get("footnotes", [])
        render_source_list(footnotes)

    # ── LIME on-demand ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Explainability (LIME)")
    agent_to_explain = st.selectbox(
        "Select agent to explain",
        ["financial", "news", "legal", "order_book", "product",
         "management", "ownership", "peer", "culture", "innovation",
         "valuation", "ratings"],
    )
    if st.button("Explain Agent Decision"):
        with st.spinner(f"Computing LIME explanation for {agent_to_explain}…"):
            try:
                ex_resp = requests.post(
                    f"{API_BASE}/explain/{job_id}/{agent_to_explain}",
                    timeout=120,
                )
                ex_resp.raise_for_status()
                ex_data = ex_resp.json()
                render_lime_panel(ex_data.get("feature_importances", []), agent_to_explain)
                st.caption(f"Local prediction: {ex_data.get('local_prediction', 0):.3f} | R²: {ex_data.get('score', 0):.3f}")
            except Exception as exc:
                st.error(f"LIME explain failed: {exc}")
