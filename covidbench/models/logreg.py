"""The honest benchmark: if this matches LGBM, the boosting is not earning its keep."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from .. import config
from ..registry import register


@register(
    "logreg",
    features=config.FEATURES_ALL,
    notes="Interpretable linear baseline",
    tracks=(config.COHORT_PAPER, config.COHORT_INCLUSIVE),
)
def logreg():
    return LogisticRegression(max_iter=1000, class_weight="balanced")
