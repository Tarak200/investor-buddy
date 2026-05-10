"""
evaluation/lime_explainer.py
------------------------------
LIME feature importance for any specialist agent's output.

How it works:
  1. Convert claims to a numeric feature vector (confidence + cleanliness per claim).
  2. Use LimeTabularExplainer with N=50 perturbations.
  3. Score each perturbed sample with a fast analytical surrogate (weighted mean).
  4. Fit a local linear model and return feature importances.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from models.sourced_claim import FeatureImportance, LIMEExplanation, SourcedClaim

log = structlog.get_logger(__name__)

_N_PERTURBATIONS = 50
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


def _llm_surrogate_score(feature_vector: np.ndarray, feature_names: list[str], agent_name: str = "") -> float:
    """
    Fast analytical surrogate: weighted mean of the feature vector.
    Confidence features (even indices) weight 0.6, cleanliness features (odd indices) weight 0.4.
    Avoids LLM calls per perturbation — the feature vector is already numeric quality scores.
    """
    if feature_vector.size == 0:
        return 0.5
    confidence_vals = feature_vector[0::2]   # every even index
    cleanliness_vals = feature_vector[1::2]  # every odd index
    c_mean = float(np.mean(confidence_vals)) if confidence_vals.size > 0 else 0.5
    cl_mean = float(np.mean(cleanliness_vals)) if cleanliness_vals.size > 0 else 0.5
    return 0.6 * c_mean + 0.4 * cl_mean


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
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            r_squared=0.0,
        )

    try:
        import lime.lime_tabular  # type: ignore
    except ImportError:
        log.error("lime_not_installed")
        return LIMEExplanation(
            agent_name=agent_name,
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            r_squared=0.0,
        )

    x, feature_names = _claims_to_feature_vector(claims)
    if x.size == 0:
        return LIMEExplanation(
            agent_name=agent_name,
            feature_importances=[],
            intercept=0.0,
            local_prediction=0.5,
            r_squared=0.0,
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
        scores = [_llm_surrogate_score(row, feature_names, agent_name) for row in batch]
        return np.array(scores)

    explanation = explainer.explain_instance(
        data_row=x,
        predict_fn=_predict_fn,
        num_features=min(_MAX_FEATURES, len(feature_names)),
        num_samples=_N_PERTURBATIONS,
    )

    local_pred = float(explanation.local_pred[0]) if explanation.local_pred is not None else 0.5
    intercept_raw = explanation.intercept if hasattr(explanation, "intercept") else {}
    intercept = float(intercept_raw.get(1, intercept_raw.get(0, 0.0))) if isinstance(intercept_raw, dict) else 0.0

    def _human_label(lime_name: str) -> str:
        # lime_name may be a discretized bucket like "0.89 < claim_0_confidence <= 0.96"
        # or a plain feature name like "claim_0_confidence"
        m = re.search(r"claim_(\d+)_(confidence|cleanliness)", lime_name)
        if not m:
            return lime_name
        idx, attr = m.group(1), m.group(2)
        attr_label = "Confidence" if attr == "confidence" else "Cleanliness"
        return f"Claim #{idx} — {attr_label}"

    feature_importances = [
        FeatureImportance(
            feature=name,
            weight=float(imp),
            direction="positive" if imp >= 0 else "negative",
            human_label=_human_label(name),
        )
        for name, imp in explanation.as_list()
    ]

    return LIMEExplanation(
        agent_name=agent_name,
        feature_importances=feature_importances,
        intercept=intercept,
        local_prediction=local_pred,
        r_squared=float(explanation.score) if hasattr(explanation, "score") else 0.0,
    )
