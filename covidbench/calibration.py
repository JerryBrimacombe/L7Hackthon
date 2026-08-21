"""Leakage-safe probability calibration for binary model scores."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Map a model's positive-class scores onto probabilities."""

    def __init__(self, method: str = "sigmoid"):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.method = method
        self._model = None

    def fit(self, probabilities, y):
        probabilities = np.asarray(probabilities, dtype=float)
        y = np.asarray(y, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("calibration data must contain both classes")
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self._model.fit(_logit(probabilities).reshape(-1, 1), y)
        else:
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(probabilities, y)
        return self

    def predict(self, probabilities) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        probabilities = np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]
        return np.asarray(self._model.predict(probabilities), dtype=float)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))