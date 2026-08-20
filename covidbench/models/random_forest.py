"""Random forest baseline inspired by notebook experiments."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from .. import config
from ..registry import register


@register("random_forest", features=config.FEATURES_ALL, notes="Notebook baseline: random forest")
def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )