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


def _table_arrays(table: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(table, key=lambda r: r["p"], reverse=True)
    return (
        np.array([r["p"] for r in rows], dtype=float),
        np.array([r["n"] for r in rows], dtype=float),
        np.array([r["pos"] for r in rows], dtype=float),
    )


def metrics_from_counts(n: np.ndarray, pos: np.ndarray, capacity: float) -> dict:
    """Recompute the ranking metrics from per-score counts alone.

    The score table is a sufficient statistic, so this returns the same numbers as
    evaluate() without needing the per-row predictions.
    """
    total_n = float(n.sum())
    total_pos = float(pos.sum())
    total_neg = total_n - total_pos
    if total_pos == 0 or total_neg == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "sensitivity_at_capacity": float("nan")}

    cum_n = np.concatenate([[0.0], np.cumsum(n)])
    cum_pos = np.concatenate([[0.0], np.cumsum(pos)])

    recall = cum_pos / total_pos
    fpr = (cum_n - cum_pos) / total_neg
    roc_auc = float(np.trapezoid(recall, fpr))

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.divide(cum_pos, cum_n, out=np.zeros_like(cum_n), where=cum_n > 0)
    # Average precision is the step-wise sum, matching sklearn rather than the trapezoid.
    pr_auc = float(np.sum(np.diff(recall) * precision[1:]))

    k = max(1, int(round(capacity * total_n)))
    room = np.clip(k - cum_n[:-1], 0.0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.divide(room, n, out=np.zeros_like(n), where=n > 0)
    sensitivity = float(np.sum(pos * share) / total_pos)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "sensitivity_at_capacity": sensitivity}


def metrics_from_score_table(table: list[dict], capacity: float = config.DEFAULT_CAPACITY) -> dict:
    _, n, pos = _table_arrays(table)
    return metrics_from_counts(n, pos, capacity)


def bootstrap_confidence_intervals(
    table: list[dict],
    capacity: float = config.DEFAULT_CAPACITY,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Percentile bootstrap intervals for the headline metrics.

    Rows sharing a score are exchangeable, so resampling group sizes multinomially
    and positives binomially is equivalent to resampling individual rows, but runs
    in a few hundred operations instead of tens of thousands.
    """
    _, n, pos = _table_arrays(table)
    total_n = int(n.sum())
    if total_n == 0:
        return {}

    rng = np.random.default_rng(seed)
    group_share = n / n.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        positive_rate = np.divide(pos, n, out=np.zeros_like(n), where=n > 0)

    draws: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "sensitivity_at_capacity": []}
    for _ in range(n_boot):
        n_draw = rng.multinomial(total_n, group_share).astype(float)
        pos_draw = rng.binomial(n_draw.astype(int), positive_rate).astype(float)
        sample = metrics_from_counts(n_draw, pos_draw, capacity)
        for key, values in draws.items():
            value = sample[key]
            if not np.isnan(value):
                values.append(value)

    intervals = {}
    for key, values in draws.items():
        if not values:
            continue
        low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        intervals[f"{key}_ci_low"] = float(low)
        intervals[f"{key}_ci_high"] = float(high)
    return intervals

