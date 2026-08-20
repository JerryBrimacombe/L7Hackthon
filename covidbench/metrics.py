"""Evaluation metrics. Single owner - every model is scored through evaluate()."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def _expected_top_k_positives(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Positives caught in the top k, averaged over random tie-breaking.

    With 8 binary features there are at most 256 distinct scores, so ties at the
    cut-off are the norm, not an edge case. Naive argsort would silently reward
    whatever order the rows happen to arrive in.
    """
    grouped = (
        pd.DataFrame({"y": y, "p": p})
        .groupby("p")["y"]
        .agg(["size", "sum"])
        .sort_index(ascending=False)
    )
    remaining, caught = k, 0.0
    for _, row in grouped.iterrows():
        if remaining <= 0:
            break
        size = int(row["size"])
        take = min(remaining, size)
        caught += float(row["sum"]) * take / size
        remaining -= take
    return caught


def sensitivity_at_capacity(y, p, capacity: float) -> float:
    """Fraction of true positives found if only `capacity` of people can be tested."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    total_positives = float(y.sum())
    if total_positives == 0:
        return float("nan")
    k = max(1, int(round(capacity * len(y))))
    return _expected_top_k_positives(y, p, k) / total_positives


def score_table(y, p) -> list[dict]:
    """Counts per distinct predicted score.

    With at most 256 possible inputs this is a few KB, yet it is a sufficient
    statistic: ROC, PR, calibration and sensitivity at any capacity can all be
    reconstructed from it exactly, with no need to store per-row predictions.
    """
    grouped = (
        pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
        .groupby("p")["y"]
        .agg(["size", "sum"])
        .sort_index(ascending=False)
    )
    return [
        {"p": float(score), "n": int(row["size"]), "pos": int(row["sum"])}
        for score, row in grouped.iterrows()
    ]


def evaluate(y, p, capacity: float = config.DEFAULT_CAPACITY) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    sensitivity = sensitivity_at_capacity(y, p, capacity)
    prevalence = float(y.mean())
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "sensitivity_at_capacity": sensitivity,
        "capacity": capacity,
        "lift_at_capacity": sensitivity / capacity if capacity else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "distinct_scores": int(np.unique(p).size),
    }
