"""
evaluation/lime_explainer.py
------------------------------
LIME feature importance for any specialist agent's output.

How it works:
  1. Convert claims to a numeric feature vector (100 features max).
  2. Use LimeTabularExplainer with N=100 perturbations.
  3. Score each perturbed sample with the LLM surrogate (lime_surrogate prompt).
  4. Fit a local linear model and return feature importances.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import structlog

from llm.prompt_manager import load_prompt
from llm.provider import get_llm_json_response
from models.sourced_claim import FeatureImportance, LIMEExplanation, SourcedClaim

log = structlog.get_logger(__name__)

_N_PERTURBATIONS = 100
_MAX_FEATURES = 20


def _claims_to_feature_vector(claims: list[SourcedClaim]) -> tuple[np.ndarray, list[str]]:
    """
    Convert a list of SourcedClaims to a flat numeric feature vector.
    Returns (vector, feature_names).
    """
    feature_names: list[str] = []
    values: list[float] = []

    for i, claim in enumerate(claims[:50]):
        # confidence
        feature_names.append(f"claim_{i}_confidence")
        values.append(float(claim.confidence))
        # hallucination score (inverted for readability: 1 = clean)
        feature_names.append(f"claim_{i}_cleanliness")
        values.append(1.0 - float(claim.hallucination_score))

    return np.array(values, dtype=float), feature_names


def _llm_surrogate_score(feature_vector: np.ndarray, feature_names: list[str]) -> float:
    """Score a single perturbed feature vector using the LLM surrogate."""
    prompt_template = load_prompt("lime_surrogate")
    feature_dict = {name: round(float(val), 4) for name, val in zip(feature_names, feature_vector)}
    prompt = prompt_template.format(features=json.dumps(feature_dict))
    messages = [
        {"role": "system", "content": "You are a scoring function. Return JSON: {\"score\": float}"},
        {"role": "user", "content": prompt},
    ]
    try:
        response = get_llm_json_response(messages, temperature=0.0, max_tokens=64)
        parsed = json.loads(response)
        return float(parsed.get("score", 0.5))
    except Exception:
        return 0.5


def explain_agent(
    agent_name: str,
    claims: list[SourcedClaim],
    company: str,
    ticker: str,
) -> LIMEExplanation:
    """
    Compute LIME explanation for an agent's output.
    Returns LIMEExplanation.
    """
    if not claims:
        return LIMEExplanation(
            agent_name=agent_name,
            company=company,
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            score=0.0,
        )

    try:
        import lime.lime_tabular  # type: ignore
    except ImportError:
        log.error("lime_not_installed")
        return LIMEExplanation(
            agent_name=agent_name,
            company=company,
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            score=0.0,
        )

    x, feature_names = _claims_to_feature_vector(claims)
    if x.size == 0:
        return LIMEExplanation(
            agent_name=agent_name,
            company=company,
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            score=0.0,
        )

    # Build training dataset (N perturbations)
    rng = np.random.default_rng(42)
    training_data = rng.normal(
        loc=np.tile(x, (_N_PERTURBATIONS, 1)),
        scale=0.1,
    ).clip(0.0, 1.0)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        mode="regression",
        random_state=42,
    )

    def _predict_fn(batch: np.ndarray) -> np.ndarray:
        scores = [_llm_surrogate_score(row, feature_names) for row in batch]
        return np.array(scores)

    explanation = explainer.explain_instance(
        data_row=x,
        predict_fn=_predict_fn,
        num_features=min(_MAX_FEATURES, len(feature_names)),
        num_samples=_N_PERTURBATIONS,
    )

    local_pred = float(explanation.local_pred[0]) if explanation.local_pred is not None else 0.5
    intercept = float(explanation.intercept[1]) if hasattr(explanation, "intercept") else 0.0

    feature_importances = [
        FeatureImportance(
            feature_name=name,
            importance=float(imp),
            description=f"Claim {re.sub(r'claim_(\\d+)_.*', r'#\\1', name)} attribute",
        )
        for name, imp in explanation.as_list()
    ]

    return LIMEExplanation(
        agent_name=agent_name,
        company=company,
        feature_importances=feature_importances,
        intercept=intercept,
        local_prediction=local_pred,
        score=explanation.score if hasattr(explanation, "score") else 0.0,
    )
