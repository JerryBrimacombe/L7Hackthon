"""Empirical positive rate per feature pattern - the Bayes-optimal ceiling.

Only 2^8 = 256 input patterns exist, so this lookup table is the best any model
can do on these features. Every other entry should be read as a percentage of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..registry import register


class PatternLookup:
    def __init__(self, features: list[str]):
        self.features = list(features)

    def fit(self, X, y):
        frame = X[self.features].copy()
        frame["_y"] = np.asarray(y)
        self.table_ = frame.groupby(self.features)["_y"].mean()
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        index = pd.MultiIndex.from_frame(X[self.features])
        p = self.table_.reindex(index).to_numpy(dtype=float)
        p = np.where(np.isnan(p), self.prior_, p)
        return np.column_stack([1.0 - p, p])


@register(
    "ceiling_lookup",
    features=config.FEATURES_ALL,
    notes="Theoretical upper bound",
    tracks=(config.COHORT_PAPER,),
)
def ceiling_lookup():
    return PatternLookup(config.FEATURES_ALL)


@register(
    "ceiling_lookup_balanced",
    features=config.FEATURES_BALANCED,
    notes="Upper bound for the 5-feature balanced model",
)
def ceiling_lookup_balanced():
    # released_lgbm_balanced only sees 5 features, so the 8-feature ceiling is not its bound.
    return PatternLookup(config.FEATURES_BALANCED)


@register(
    "ceiling_lookup_inclusive",
    features=config.FEATURES_INCLUSIVE,
    notes="Upper bound for the inclusive 10-feature track",
    tracks=(config.COHORT_INCLUSIVE,),
)
def ceiling_lookup_inclusive():
    # 1024 patterns over a different population: not comparable to the 8-feature ceiling.
    return PatternLookup(config.FEATURES_INCLUSIVE)
