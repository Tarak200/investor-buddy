"""
agents/explainability_agent.py
--------------------------------
LIMEExplainabilityAgent — on-demand only, never called in the main pipeline.
Triggered exclusively via POST /api/v1/explain/{job_id}/{agent_name}.
"""

from __future__ import annotations

import structlog

from evaluation.lime_explainer import explain_agent
from models.sourced_claim import LIMEExplanation, SourcedClaim

log = structlog.get_logger(__name__)


def run(
    agent_name: str,
    claims: list[SourcedClaim],
    company: str,
    ticker: str,
) -> LIMEExplanation:
    """
    Compute LIME feature importances for the given agent's output.
    Returns a LIMEExplanation object.
    """
    log.info("lime_explain_start", agent=agent_name, company=company)
    explanation = explain_agent(agent_name=agent_name, claims=claims, company=company, ticker=ticker)
    log.info("lime_explain_done", agent=agent_name, features=len(explanation.feature_importances))
    return explanation
