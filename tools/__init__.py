"""
tools/__init__.py
-----------------
Public API: re-exports every tool function used by the agents.
"""

from tools._base import make_claim, safe_get, random_headers
from tools.financial_tools import (
    get_balance_sheet,
    get_cashflow_statement,
    get_eps_history,
    get_key_ratios,
    get_pl_statement,
    get_quarterly_results,
    get_annual_report_data,
)
from tools.news_tools import get_recent_news, analyze_sentiment
from tools.legal_tools import (
    check_sebi_enforcement,
    check_mca_filings,
    search_legal_news,
)
from tools.tender_tools import search_government_tenders, get_order_book_data
from tools.management_tools import (
    get_management_profiles,
    get_employee_reviews,
    get_board_qualifications,
)
from tools.web_search_tools import (
    analyze_products,
    get_industry_outlook,
    get_market_share,
    get_product_segment_revenue,
)
from tools.ownership_tools import (
    get_promoter_holding,
    get_promoter_pledging,
    get_institutional_ownership,
    get_top_shareholders,
    get_ownership_trend,
)
from tools.peer_tools import (
    get_peer_list,
    get_peer_financials,
    get_price_comparison,
    get_product_price_benchmarking,
    get_peer_market_share_comparison,
)
from tools.culture_tools import (
    get_reddit_employee_sentiment,
    get_twitter_employee_chatter,
    get_glassdoor_culture,
    get_ambitionbox_culture,
    get_new_project_signals,
    aggregate_culture_signal,
)
from tools.valuation_tools import (
    get_intrinsic_value_dcf,
    get_relative_valuation,
    get_graham_number,
    get_peg_ratio,
    compute_valuation_verdict,
    get_price_technicals,
    get_technical_verdict,
    get_new_verticals,
    estimate_vertical_eps_impact,
)
from tools.ratings_tools import (
    get_broker_recommendations,
    get_credit_ratings,
    get_esg_scores,
    get_index_memberships,
    get_mutual_fund_holdings,
    get_government_schemes_benefit,
)
from tools.innovation_tools import (
    get_rd_spending,
    get_patent_activity,
    get_global_presence,
    get_geographic_revenue_split,
    get_international_subsidiaries,
    get_export_trends,
    get_competition_rankings,
    get_certifications,
)
from tools.forecast_tools import (
    get_time_horizon_weights,
    compute_eps_projections,
    compute_revenue_projections,
    compute_price_projections,
    compute_market_share_projections,
    synthesize_investment_thesis,
)
from tools.chart_tools import (
    create_financial_charts,
    create_eps_growth_chart,
    create_sentiment_chart,
    create_order_pipeline_chart,
    create_management_radar,
    create_risk_radar,
    create_ownership_chart,
    create_peer_comparison_chart,
    create_market_share_chart,
    create_culture_charts,
    create_innovation_charts,
    create_valuation_charts,
    create_ratings_charts,
    create_forecast_charts,
)

__all__ = [
    # base helpers
    "make_claim", "safe_get", "random_headers",
    # financial
    "get_balance_sheet", "get_cashflow_statement", "get_eps_history",
    "get_key_ratios", "get_pl_statement", "get_quarterly_results", "get_annual_report_data",
    # news
    "get_recent_news", "analyze_sentiment",
    # legal
    "check_sebi_enforcement", "check_mca_filings", "search_legal_news",
    # tender / order book
    "search_government_tenders", "get_order_book_data",
    # management
    "get_management_profiles", "get_employee_reviews", "get_board_qualifications",
    # web / product
    "analyze_products", "get_industry_outlook", "get_market_share", "get_product_segment_revenue",
    # ownership
    "get_promoter_holding", "get_promoter_pledging", "get_institutional_ownership",
    "get_top_shareholders", "get_ownership_trend",
    # peer
    "get_peer_list", "get_peer_financials", "get_price_comparison",
    "get_product_price_benchmarking", "get_peer_market_share_comparison",
    # culture
    "get_reddit_employee_sentiment", "get_twitter_employee_chatter",
    "get_glassdoor_culture", "get_ambitionbox_culture",
    "get_new_project_signals", "aggregate_culture_signal",
    # valuation / technical
    "get_intrinsic_value_dcf", "get_relative_valuation", "get_graham_number",
    "get_peg_ratio", "compute_valuation_verdict", "get_price_technicals",
    "get_technical_verdict", "get_new_verticals", "estimate_vertical_eps_impact",
    # ratings
    "get_broker_recommendations", "get_credit_ratings", "get_esg_scores",
    "get_index_memberships", "get_mutual_fund_holdings", "get_government_schemes_benefit",
    # innovation / global
    "get_rd_spending", "get_patent_activity", "get_global_presence",
    "get_geographic_revenue_split", "get_international_subsidiaries",
    "get_export_trends", "get_competition_rankings", "get_certifications",
    # forecast
    "get_time_horizon_weights", "compute_eps_projections", "compute_revenue_projections",
    "compute_price_projections", "compute_market_share_projections", "synthesize_investment_thesis",
    # charts
    "create_financial_charts", "create_eps_growth_chart", "create_sentiment_chart",
    "create_order_pipeline_chart", "create_management_radar", "create_risk_radar",
    "create_ownership_chart", "create_peer_comparison_chart", "create_market_share_chart",
    "create_culture_charts", "create_innovation_charts", "create_valuation_charts",
    "create_ratings_charts", "create_forecast_charts",
]
