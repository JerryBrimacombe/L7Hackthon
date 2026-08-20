"""The published covidpred artifacts, used as the reference baseline."""
from __future__ import annotations

import numpy as np

from .. import config
from ..registry import register


class ReleasedBooster:
    pretrained = True

    def __init__(self, filename: str, expected_names: list[str], features: list[str]):
        import lightgbm as lgb

        path = config.covidpred_root() / filename
        # LightGBM splits trees using the byte offsets in tree_sizes, so a Windows
        # checkout with core.autocrlf=true corrupts the file. Normalise to LF.
        text = path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
        self.booster = lgb.Booster(model_str=text)
        actual = self.booster.feature_name()
        if actual != expected_names:
            raise AssertionError(f"Feature order mismatch: {actual} != {expected_names}")
        self.features = features

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # Passed positionally: our column names differ from the dump's "Age_60+".
        p = self.booster.predict(X[self.features].to_numpy(dtype=float))
        return np.column_stack([1.0 - p, p])


@register("released_lgbm_all", features=config.FEATURES_ALL, notes="Published 8-feature artifact")
def released_all():
    return ReleasedBooster(
        "lgbm_model_all_features.txt", config.RELEASED_ALL_NAMES, config.FEATURES_ALL
    )


@register(
    "released_lgbm_balanced",
    features=config.FEATURES_BALANCED,
    notes="Published 5-feature artifact",
)
def released_balanced():
    return ReleasedBooster(
        "lgbm_model_balanced_features.txt",
        config.RELEASED_BALANCED_NAMES,
        config.FEATURES_BALANCED,
    )
