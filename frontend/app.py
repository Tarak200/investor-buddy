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

import os
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

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

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

    st.divider()
    st.header("🔭 Discover Interesting Stocks")
    discover_market = st.selectbox("Market to Discover", ["INDIA", "US"], key="discover_market")

    _INDIA_SECTORS = [
        "All Sectors",
        "Automobile",
        "Banking & Finance",
        "Chemicals",
        "Consumer Durables",
        "Defence",
        "Energy & Power",
        "FMCG",
        "Healthcare",
        "Infrastructure",
        "IT & Technology",
        "Media & Entertainment",
        "Metals & Mining",
        "Pharmaceuticals",
        "Real Estate",
        "Telecom",
        "Textiles",
    ]
    _US_SECTORS = [
        "All Sectors",
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Healthcare",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    ]
    _sector_options = _INDIA_SECTORS if discover_market == "INDIA" else _US_SECTORS
    discover_sector = st.selectbox("Sector", _sector_options, key="discover_sector")

    _MARKET_CAP_OPTIONS = ["Micro Cap", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    discover_market_caps = st.multiselect(
        "Market Cap",
        _MARKET_CAP_OPTIONS,
        default=[],
        key="discover_market_caps",
        placeholder="Any market cap (leave blank for all)",
    )

    discover_btn = st.button("✨ Discover Stocks", use_container_width=True)

# ── Trigger Discover ──────────────────────────────────────────────────────────
if discover_btn:
    _sector_payload = None if discover_sector == "All Sectors" else discover_sector
    _caps_payload = discover_market_caps if discover_market_caps else None
    _filter_summary = []
    if _sector_payload:
        _filter_summary.append(f"Sector: {_sector_payload}")
    if _caps_payload:
        _filter_summary.append(f"Market Cap: {', '.join(_caps_payload)}")
    _spinner_label = f"Queuing discovery job for {discover_market}" + (f" ({', '.join(_filter_summary)})" if _filter_summary else "") + "…"
    with st.spinner(_spinner_label):
        try:
            d_resp = requests.post(
                f"{API_BASE}/discover",
                json={
                    "market": discover_market,
                    "sector": _sector_payload,
                    "market_caps": _caps_payload,
                },
                timeout=10,
            )
            d_resp.raise_for_status()
            d_job_id = d_resp.json()["job_id"]
            st.session_state["discover_job_id"] = d_job_id
            st.session_state["discover_result"] = None
            st.success(f"Discovery job queued: `{d_job_id}`")
        except Exception as exc:
            st.error(f"Failed to start discovery: {exc}")

# ── Poll Discovery ─────────────────────────────────────────────────────────────
d_job_id: str | None = st.session_state.get("discover_job_id")
discover_result: dict | None = st.session_state.get("discover_result")

if d_job_id and not discover_result:
    d_status_ph = st.empty()
    d_progress = st.progress(0)
    max_d_polls = 300        # 15 minutes max (300 × 3s)
    d_polls = 0
    while d_polls < max_d_polls:
        d_polls += 1
        try:
            ds_resp = requests.get(f"{API_BASE}/discover/{d_job_id}", timeout=5)
            if ds_resp.status_code == 202:
                d_status_ph.info("Discovery in progress… scanning superstar portfolios & govt schemes")
                d_progress.progress(min(d_polls / max_d_polls, 0.95))
                time.sleep(3)
                continue
            if ds_resp.status_code == 200:
                d_progress.progress(1.0)
                st.session_state["discover_result"] = ds_resp.json()
                discover_result = st.session_state["discover_result"]
                d_status_ph.success("Discovery complete!")
                break
            if ds_resp.status_code == 500:
                d_status_ph.error(f"Discovery job failed: {ds_resp.json().get('detail', ds_resp.text)}")
                st.session_state["discover_job_id"] = None
                break
            ds_resp.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 202:
                d_status_ph.info("Discovery in progress…")
                d_progress.progress(min(d_polls / max_d_polls, 0.95))
                time.sleep(3)
                continue
            d_status_ph.error(f"Discovery failed: {exc}")
            st.session_state["discover_job_id"] = None
            break
        except Exception:
            time.sleep(3)
            continue
    else:
        st.warning("Discovery is taking longer than expected. Click **Discover Stocks** again to retry.")
        st.session_state["discover_job_id"] = None

if discover_result:
    _d_market = discover_result.get('market', '')
    _d_sector = discover_result.get('sector')
    _d_caps = discover_result.get('market_caps')
    _d_title_parts = [f"🔭 Discovered Stocks — {_d_market}"]
    if _d_sector:
        _d_title_parts.append(_d_sector)
    if _d_caps:
        _d_title_parts.append(", ".join(_d_caps))
    st.subheader(" | ".join(_d_title_parts))
    top_picks = discover_result.get("top_picks", [])
    if not top_picks:
        st.info("No top picks found in this run.")
    else:
        for pick in top_picks:
            with st.expander(f"**{pick.get('company', '')}** ({pick.get('ticker', '')}) — Score: {pick.get('composite_score', 0):.2f}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Superstar Conviction", f"{pick.get('superstar_conviction', 0):.2f}")
                col2.metric("Policy Tailwind", f"{pick.get('policy_tailwind', 0):.2f}")
                col3.metric("Composite Score", f"{pick.get('composite_score', 0):.2f}")
                st.markdown(f"**Sector:** {pick.get('sector', 'N/A')}")
                st.markdown(f"**Rationale:** {pick.get('rationale', 'N/A')}")
                investors = pick.get("investors_backing", [])
                if investors:
                    st.markdown(f"**Investors Backing:** {', '.join(investors)}")
                catalysts = pick.get("policy_catalysts", [])
                if catalysts:
                    st.markdown("**Policy Catalysts:**")
                    for cat in catalysts:
                        st.markdown(f"- {cat}")
    elapsed = discover_result.get("elapsed_seconds", 0)
    evaluated = discover_result.get("candidates_evaluated", 0)
    st.caption(f"Evaluated {evaluated} candidates in {elapsed:.1f}s")

st.divider()

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

        if s_data.get("status") == "failed":
            errors = s_data.get("errors") or []
            error_detail = errors[0] if errors else "Unknown error"
            progress_bar.empty()
            status_placeholder.error(f"Analysis failed: {error_detail}")
            st.session_state["job_id"] = None
            break

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
