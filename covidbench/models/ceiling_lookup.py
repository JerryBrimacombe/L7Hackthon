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


@register("ceiling_lookup", features=config.FEATURES_ALL, notes="Theoretical upper bound")
def ceiling_lookup():
    return PatternLookup(config.FEATURES_ALL)
