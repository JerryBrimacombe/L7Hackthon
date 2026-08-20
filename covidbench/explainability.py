"""Model explainability helpers for leaderboard reporting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplainabilitySummary:
    method: str
    top_features: list[dict]


def _unwrap_estimator(model):
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model


def summarize(model, feature_names: list[str], top_k: int = 8) -> dict | None:
    """Return a compact, serializable explanation summary when available."""
    estimator = _unwrap_estimator(model)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2 and coef.shape[0] == 1 and coef.shape[1] == len(feature_names):
            values = coef[0]
            order = np.argsort(np.abs(values))[::-1][:top_k]
            return ExplainabilitySummary(
                method="coefficients",
                top_features=[
                    {
                        "feature": feature_names[idx],
                        "value": float(values[idx]),
                        "abs_value": float(abs(values[idx])),
                    }
                    for idx in order
                ],
            ).__dict__

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        if values.ndim == 1 and values.shape[0] == len(feature_names):
            order = np.argsort(values)[::-1][:top_k]
            return ExplainabilitySummary(
                method="feature_importance",
                top_features=[
                    {
                        "feature": feature_names[idx],
                        "value": float(values[idx]),
                    }
                    for idx in order
                ],
            ).__dict__

    return None