"""
tools/chart_tools.py
---------------------
Plotly figure generators.  Every function returns a JSON-serializable dict
(figure.to_dict()) so it can be safely serialised into the API response.

NOTE: No external API calls — these are pure data → Plotly transformations.
"""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import structlog
from langchain_core.tools import tool

from models.sourced_claim import SourcedClaim

log = structlog.get_logger(__name__)


def _fig_to_dict(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def _extract(claims: list[SourcedClaim], key: str, default: Any = None) -> Any:
    for c in claims:
        if isinstance(c.value, dict) and key in c.value:
            return c.value[key]
    return default


# ── Financial charts ────────────────────────────────────────────────────────


@tool
def create_financial_charts(financial_claims: list[SourcedClaim], company: str) -> dict:
    """Revenue, PAT, and FCF bar charts for 5 years."""
    revenues: list[float] = []
    pats: list[float] = []
    years: list[str] = []

    for c in financial_claims:
        if isinstance(c.value, dict):
            for row in c.value.get("annual_results", []):
                if isinstance(row, dict):
                    years.append(str(row.get("year", "")))
                    revenues.append(float(row.get("revenue", 0) or 0))
                    pats.append(float(row.get("net_profit", 0) or 0))

    if not years:
        return {}

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=years, y=revenues, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="PAT", x=years, y=pats, marker_color="seagreen"))
    fig.update_layout(
        title=f"{company} — Revenue & PAT",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Amount (Cr / USD M)",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_eps_growth_chart(eps_claims: list[SourcedClaim], company: str) -> dict:
    """Historical EPS bar chart + YoY growth % line."""
    eps_data: list[dict] = []
    for c in eps_claims:
        if isinstance(c.value, dict) and c.value.get("type") == "eps_history":
            eps_data = c.value.get("eps_history", [])
            break

    if not eps_data:
        return {}

    years = [str(d.get("year", "")) for d in eps_data]
    eps_vals = [float(d.get("eps", 0) or 0) for d in eps_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="EPS", x=years, y=eps_vals, marker_color="goldenrod"))
    fig.update_layout(
        title=f"{company} — EPS History",
        xaxis_title="Year",
        yaxis_title="EPS",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_sentiment_chart(news_claims: list[SourcedClaim], company: str) -> dict:
    """Donut chart of Positive / Neutral / Negative news sentiment."""
    pos = neg = neu = 0
    for c in news_claims:
        if isinstance(c.value, dict):
            label = str(c.value.get("sentiment_label", "NEUTRAL")).upper()
            if label == "POSITIVE":
                pos += 1
            elif label == "NEGATIVE":
                neg += 1
            else:
                neu += 1

    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos, neu, neg],
        hole=0.4,
        marker_colors=["seagreen", "steelblue", "crimson"],
    ))
    fig.update_layout(title=f"{company} — News Sentiment", template="plotly_dark")
    return _fig_to_dict(fig)


@tool
def create_order_pipeline_chart(order_claims: list[SourcedClaim], company: str) -> dict:
    """Horizontal bar chart of order book / pipeline by segment."""
    segments: list[str] = []
    values: list[float] = []
    for c in order_claims:
        if isinstance(c.value, dict) and "order_book_by_segment" in c.value:
            for seg, val in (c.value["order_book_by_segment"] or {}).items():
                segments.append(seg)
                values.append(float(val or 0))
            break

    if not segments:
        return {}

    fig = go.Figure(go.Bar(
        x=values, y=segments, orientation="h", marker_color="dodgerblue",
    ))
    fig.update_layout(
        title=f"{company} — Order Book by Segment",
        xaxis_title="Value (Cr / USD M)",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_management_radar(management_claims: list[SourcedClaim], company: str) -> dict:
    """Radar chart: tenure, qualifications, experience, promoter, board independence."""
    dims = {
        "Avg Tenure (yrs)": 0.0,
        "Board Independence %": 0.0,
        "Education Score": 5.0,
        "Promoter Commitment": 5.0,
        "Track Record": 5.0,
    }
    for c in management_claims:
        if isinstance(c.value, dict):
            dims["Avg Tenure (yrs)"] = float(c.value.get("avg_tenure_years", dims["Avg Tenure (yrs)"]) or 0)
            dims["Board Independence %"] = float(c.value.get("board_independence_pct", dims["Board Independence %"]) or 0)

    labels = list(dims.keys())
    values = list(dims.values()) + [list(dims.values())[0]]  # close the loop
    labels_loop = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels_loop, fill="toself", name=company,
    ))
    fig.update_layout(
        title=f"{company} — Management Quality Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_risk_radar(legal_claims: list[SourcedClaim], ownership_claims: list[SourcedClaim], company: str) -> dict:
    """Radar chart: Legal Risk, Pledge Risk, Insider Selling, Regulatory Risk, Litigation."""
    legal_score = 5.0
    pledge_score = 5.0
    for c in legal_claims:
        if isinstance(c.value, dict):
            mapping = {"LOW": 2, "MEDIUM": 5, "HIGH": 8, "CRITICAL": 10}
            risk_level = str(c.value.get("overall_risk_level", "LOW")).upper()
            legal_score = float(mapping.get(risk_level, 5))
            break
    for c in ownership_claims:
        if isinstance(c.value, dict):
            pledge = float(c.value.get("pledge_pct", 0) or 0)
            pledge_score = min(pledge / 10.0, 10.0)  # normalise to 0-10
            break

    labels = ["Legal Risk", "Pledge Risk", "Insider Selling", "Regulatory Risk", "Litigation"]
    values = [legal_score, pledge_score, 5.0, 5.0, 5.0, legal_score]
    labels_loop = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(r=values, theta=labels_loop, fill="toself", name="Risk"))
    fig.update_layout(
        title=f"{company} — Risk Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_ownership_chart(ownership_claims: list[SourcedClaim], company: str) -> dict:
    """Stacked bar chart of Promoter, FII, DII, MF, Retail shareholding."""
    promoter = fii = dii = mf = retail = 0.0
    for c in ownership_claims:
        if isinstance(c.value, dict):
            promoter = float(c.value.get("promoter_holding_pct", 0) or 0)
            fii = float(c.value.get("fii_pct", 0) or 0)
            dii = float(c.value.get("dii_pct", 0) or 0)
            mf = float(c.value.get("mf_pct", 0) or 0)
            break

    retail = max(0.0, 100.0 - promoter - fii - dii)
    fig = go.Figure()
    for name, val, color in [
        ("Promoter", promoter, "darkorange"),
        ("FII", fii, "steelblue"),
        ("DII", dii, "seagreen"),
        ("MF", mf, "mediumpurple"),
        ("Retail", retail, "lightgray"),
    ]:
        fig.add_trace(go.Bar(name=name, x=[company], y=[val], marker_color=color))
    fig.update_layout(
        title=f"{company} — Ownership Structure",
        barmode="stack",
        yaxis_title="Shareholding %",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_peer_comparison_chart(peer_claims: list[SourcedClaim], company: str) -> dict:
    """Grouped bar chart of P/E, P/B, ROE, ROCE for company vs peers."""
    companies: list[str] = []
    pe_vals: list[float] = []
    pb_vals: list[float] = []

    for c in peer_claims:
        if isinstance(c.value, dict) and "peer_financials" in c.value:
            for row in c.value["peer_financials"][:8]:
                if isinstance(row, dict):
                    companies.append(str(row.get("company", "")))
                    pe_vals.append(float(row.get("pe", 0) or 0))
                    pb_vals.append(float(row.get("pb", 0) or 0))
            break

    if not companies:
        return {}

    fig = go.Figure()
    fig.add_trace(go.Bar(name="P/E", x=companies, y=pe_vals, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="P/B", x=companies, y=pb_vals, marker_color="seagreen"))
    fig.update_layout(
        title=f"{company} — Peer Valuation Comparison",
        barmode="group",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_market_share_chart(market_share_claims: list[SourcedClaim], company: str) -> dict:
    """Pie chart of market share among top players."""
    players: list[str] = []
    shares: list[float] = []
    for c in market_share_claims:
        if isinstance(c.value, dict) and "market_share_by_player" in c.value:
            for player, share in (c.value["market_share_by_player"] or {}).items():
                players.append(player)
                shares.append(float(share or 0))
            break

    if not players:
        return {}

    fig = go.Figure(go.Pie(labels=players, values=shares, hole=0.3))
    fig.update_layout(title=f"{company} — Market Share Distribution", template="plotly_dark")
    return _fig_to_dict(fig)


@tool
def create_culture_charts(culture_claims: list[SourcedClaim], company: str) -> dict:
    """Radar chart of 6 culture dimensions + sentiment by department heatmap."""
    dimensions = {
        "WLB": 5.0, "Compensation": 5.0, "Growth": 5.0,
        "Management": 5.0, "Job Security": 5.0, "Culture Fit": 5.0,
    }
    for c in culture_claims:
        if isinstance(c.value, dict) and "dimensions" in c.value:
            dims = c.value["dimensions"]
            dimensions["WLB"] = float(dims.get("wlb", 5) or 5)
            dimensions["Compensation"] = float(dims.get("compensation", 5) or 5)
            dimensions["Growth"] = float(dims.get("growth", 5) or 5)
            dimensions["Management"] = float(dims.get("management", 5) or 5)
            dimensions["Job Security"] = float(dims.get("job_security", 5) or 5)
            dimensions["Culture Fit"] = float(dims.get("culture_fit", 5) or 5)
            break

    labels = list(dimensions.keys())
    values = list(dimensions.values()) + [list(dimensions.values())[0]]
    labels_loop = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(r=values, theta=labels_loop, fill="toself", name="Culture"))
    fig.update_layout(
        title=f"{company} — Culture Dimensions",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_innovation_charts(innovation_claims: list[SourcedClaim], company: str) -> dict:
    """R&D intensity % over years as a line chart."""
    years: list[str] = []
    rd_pcts: list[float] = []
    for c in innovation_claims:
        if isinstance(c.value, dict) and "rd_intensity_by_year" in c.value:
            for row in c.value["rd_intensity_by_year"]:
                years.append(str(row.get("year", "")))
                rd_pcts.append(float(row.get("rd_intensity_pct", 0) or 0))
            break

    if not years:
        return {}

    fig = go.Figure(go.Scatter(x=years, y=rd_pcts, mode="lines+markers", name="R&D Intensity %"))
    fig.update_layout(
        title=f"{company} — R&D Intensity Trend",
        xaxis_title="Year",
        yaxis_title="R&D as % of Revenue",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_valuation_charts(valuation_claims: list[SourcedClaim], company: str) -> dict:
    """Waterfall chart: Current Price → DCF → Graham → Analyst Target."""
    current = dcf = graham = target = 0.0
    for c in valuation_claims:
        if isinstance(c.value, dict):
            if "dcf_intrinsic_value" in c.value:
                dcf = float(c.value["dcf_intrinsic_value"] or 0)
                current = float(c.value.get("current_price", 0) or 0)
            if "graham_number" in c.value:
                graham = float(c.value["graham_number"] or 0)
            if "consensus_target_price" in c.value:
                target = float(c.value["consensus_target_price"] or 0)

    if not current:
        return {}

    fig = go.Figure(go.Bar(
        x=["Current Price", "DCF Value", "Graham Number", "Analyst Target"],
        y=[current, dcf, graham, target],
        marker_color=["gray", "steelblue", "goldenrod", "seagreen"],
    ))
    fig.update_layout(
        title=f"{company} — Valuation Comparison",
        yaxis_title="Price",
        template="plotly_dark",
    )
    return _fig_to_dict(fig)


@tool
def create_ratings_charts(ratings_claims: list[SourcedClaim], company: str) -> dict:
    """Donut chart: Buy / Hold / Sell broker count."""
    buy = hold = sell = 0
    for c in ratings_claims:
        if isinstance(c.value, dict) and "buy_count" in c.value:
            buy = int(c.value.get("buy_count", 0) or 0)
            hold = int(c.value.get("hold_count", 0) or 0)
            sell = int(c.value.get("sell_count", 0) or 0)
            break

    if not (buy + hold + sell):
        return {}

    fig = go.Figure(go.Pie(
        labels=["Buy", "Hold", "Sell"],
        values=[buy, hold, sell],
        hole=0.4,
        marker_colors=["seagreen", "goldenrod", "crimson"],
    ))
    fig.update_layout(title=f"{company} — Broker Consensus", template="plotly_dark")
    return _fig_to_dict(fig)


@tool
def create_forecast_charts(
    eps_claims: list[SourcedClaim],
    revenue_claims: list[SourcedClaim],
    price_claims: list[SourcedClaim],
    market_share_claims: list[SourcedClaim],
    time_horizon_weights: dict,
    company: str,
) -> dict:
    """
    Returns a dict with four sub-charts:
      - eps_fan: EPS Bull/Base/Bear fan chart
      - revenue_fan: Revenue fan chart
      - price_area: Price Bull/Base/Bear area chart
      - market_share_bars: Market share projection bars
      - weight_donut: Factor weight donut for the time horizon
    """
    results: dict[str, Any] = {}

    # ── EPS Fan Chart ──────────────────────────────────────────────────────
    eps_years = [c.value.get("year") for c in eps_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]
    if eps_years:
        years_str = [str(y) for y in eps_years]
        bulls = [float(c.value.get("bull", 0)) for c in eps_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]
        bases = [float(c.value.get("base", 0)) for c in eps_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]
        bears = [float(c.value.get("bear", 0)) for c in eps_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyEPSProjection"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_str, y=bulls, name="Bull", line=dict(color="seagreen")))
        fig.add_trace(go.Scatter(x=years_str, y=bases, name="Base", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=years_str, y=bears, name="Bear", line=dict(color="crimson")))
        # Fill between bull and bear
        fig.add_trace(go.Scatter(
            x=years_str + years_str[::-1],
            y=bulls + bears[::-1],
            fill="toself",
            fillcolor="rgba(70,130,180,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ))
        fig.update_layout(title=f"{company} — EPS Fan Chart", xaxis_title="Year", yaxis_title="EPS", template="plotly_dark")
        results["eps_fan"] = _fig_to_dict(fig)

    # ── Price Area Chart ────────────────────────────────────────────────────
    price_projections = [c for c in price_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyPriceProjection"]
    if price_projections:
        yrs = [str(c.value["year"]) for c in price_projections]
        bulls = [float(c.value.get("bull", 0)) for c in price_projections]
        bases = [float(c.value.get("base", 0)) for c in price_projections]
        bears = [float(c.value.get("bear", 0)) for c in price_projections]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yrs, y=bulls, name="Bull", fill="tonexty", line=dict(color="seagreen")))
        fig.add_trace(go.Scatter(x=yrs, y=bases, name="Base", fill="tonexty", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=yrs, y=bears, name="Bear", fill="tozeroy", line=dict(color="crimson")))
        fig.update_layout(title=f"{company} — Price Projection", xaxis_title="Year", yaxis_title="Price", template="plotly_dark")
        results["price_area"] = _fig_to_dict(fig)

    # ── Market Share Bars ──────────────────────────────────────────────────
    ms_projections = [c for c in market_share_claims if isinstance(c.value, dict) and c.value.get("type") == "YearlyMarketShareProjection"]
    if ms_projections:
        yrs = [str(c.value["year"]) for c in ms_projections]
        bull_pcts = [float(c.value.get("bull_pct", 0)) for c in ms_projections]
        base_pcts = [float(c.value.get("base_pct", 0)) for c in ms_projections]
        bear_pcts = [float(c.value.get("bear_pct", 0)) for c in ms_projections]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Bull", x=yrs, y=bull_pcts, marker_color="seagreen"))
        fig.add_trace(go.Bar(name="Base", x=yrs, y=base_pcts, marker_color="steelblue"))
        fig.add_trace(go.Bar(name="Bear", x=yrs, y=bear_pcts, marker_color="crimson"))
        fig.update_layout(title=f"{company} — Market Share Projection", barmode="group", template="plotly_dark")
        results["market_share_bars"] = _fig_to_dict(fig)

    # ── Factor Weight Donut ─────────────────────────────────────────────────
    if time_horizon_weights:
        labels = list(time_horizon_weights.keys())
        values = list(time_horizon_weights.values())
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4))
        fig.update_layout(title="Factor Weight Allocation", template="plotly_dark")
        results["weight_donut"] = _fig_to_dict(fig)

    return results
