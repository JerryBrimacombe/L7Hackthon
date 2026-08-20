"""Retrained from the hyperparameters published in covidpred/hyperparameters.txt."""
from __future__ import annotations

from lightgbm import LGBMClassifier

from .. import config
from ..registry import register


@register("lgbm_retrained", features=config.FEATURES_ALL, notes="Paper hyperparameters, retrained")
def lgbm_retrained():
    return LGBMClassifier(
        num_leaves=20,
        min_child_samples=4,
        colsample_bytree=0.2,
        subsample=0.8,
        subsample_freq=5,
        learning_rate=0.05,
        n_estimators=603,
        is_unbalance=True,
        random_state=config.RANDOM_SEED,
        verbose=-1,
    )
