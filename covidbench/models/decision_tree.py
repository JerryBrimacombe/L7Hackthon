"""Decision tree baseline inspired by notebook experiments."""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from .. import config
from ..registry import register


@register("decision_tree", features=config.FEATURES_ALL, notes="Notebook baseline: decision tree")
def decision_tree():
    return DecisionTreeClassifier(
        random_state=config.RANDOM_SEED,
        class_weight="balanced",
        min_samples_leaf=8,
    )