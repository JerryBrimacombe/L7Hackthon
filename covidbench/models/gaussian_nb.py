"""Gaussian Naive Bayes baseline inspired by notebook experiments."""
from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

from .. import config
from ..registry import register


@register("gaussian_nb", features=config.FEATURES_ALL, notes="Notebook baseline: Gaussian Naive Bayes")
def gaussian_nb():
    # Kept intentionally simple as a high-bias baseline.
    _ = config.RANDOM_SEED
    return GaussianNB()