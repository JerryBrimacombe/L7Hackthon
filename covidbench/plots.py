"""Chart rendering for the leaderboard.

Kept separate from metrics.py and the model files so charting work and model work
never touch the same file. Every chart is reconstructed from the `score_table`
recorded with each result, so no per-row predictions are needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CI runners have no display

import matplotlib.pyplot as plt
import numpy as np

from . import config

CEILING_MODEL = "ceiling_lookup"
DPI = 140


def _curve(score_table: list[dict]) -> dict:
    """Cumulative statistics walking from the highest score downwards."""
    rows = sorted(score_table, key=lambda r: r["p"], reverse=True)
    n = np.array([r["n"] for r in rows], dtype=float)
    pos = np.array([r["pos"] for r in rows], dtype=float)

    total_n = n.sum()
    total_pos = pos.sum()
    total_neg = total_n - total_pos

    cum_n = np.concatenate([[0.0], np.cumsum(n)])
    cum_pos = np.concatenate([[0.0], np.cumsum(pos)])

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.divide(cum_pos, cum_n, out=np.full_like(cum_n, np.nan), where=cum_n > 0)

    return {
        "capacity": cum_n / total_n,
        "sensitivity": cum_pos / total_pos,
        "precision": precision,
        "fpr": (cum_n - cum_pos) / total_neg,
        "p": np.array([r["p"] for r in rows], dtype=float),
        "n": n,
        "rate": np.divide(pos, n, out=np.zeros_like(n), where=n > 0),
    }


def _colours(names) -> dict:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(sorted(names))}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _reliability(score_table: list[dict], bins: int = 10):
    """Equal-mass reliability curve: each bin holds a similar number of people."""
    rows = sorted(score_table, key=lambda r: r["p"])
    p = np.array([r["p"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    pos = np.array([r["pos"] for r in rows], dtype=float)

    cumulative = np.cumsum(n)
    edges = np.linspace(0, cumulative[-1], bins + 1)
    assigned = np.clip(np.searchsorted(edges[1:-1], cumulative - n / 2, side="right"), 0, bins - 1)

    predicted, observed, weight = [], [], []
    for b in range(bins):
        mask = assigned == b
        total = n[mask].sum()
        if total == 0:
            continue
        predicted.append(float((p[mask] * n[mask]).sum() / total))
        observed.append(float(pos[mask].sum() / total))
        weight.append(float(total))
    return np.array(predicted), np.array(observed), np.array(weight)


def sensitivity_vs_capacity(
    payloads: list[dict], path: Path, ceiling_model: str = CEILING_MODEL
) -> Path:
    colours = _colours(p["model"] for p in payloads)
    fig, (full_ax, zoom_ax) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    capacity = payloads[0]["metrics"]["capacity"]

    for ax in (full_ax, zoom_ax):
        for payload in sorted(payloads, key=lambda p: p["model"]):
            name = payload["model"]
            curve = _curve(payload["score_table"])
            is_ceiling = name == ceiling_model
            ax.plot(
                curve["capacity"],
                curve["sensitivity"],
                label=name,
                color="black" if is_ceiling else colours[name],
                linestyle="--" if is_ceiling else "-",
                linewidth=2.2 if is_ceiling else 1.3,
                alpha=0.45 if is_ceiling else 0.95,
                # Ceiling sits behind: the models cluster on top of it.
                zorder=1 if is_ceiling else 2,
            )
        ax.axvline(capacity, color="crimson", linewidth=0.9, alpha=0.7)

    full_ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle=":", label="random")
    _style(
        full_ax,
        "Sensitivity vs testing capacity",
        "Fraction of people tested (capacity)",
        "Fraction of cases caught (sensitivity)",
    )
    full_ax.set_xlim(0, 1)
    full_ax.set_ylim(0, 1.02)
    full_ax.annotate(
        f"reported capacity ({capacity:.0%})",
        xy=(capacity + 0.03, 0.06),
        fontsize=8,
        color="crimson",
    )
    full_ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    _style(zoom_ax, "Detail: the operating region", "Capacity", "Sensitivity")
    zoom_ax.set_xlim(0, 0.3)
    zoom_ax.set_ylim(0, 0.95)
    return _save(fig, path)


def ceiling_bars(payloads: list[dict], path: Path, ceiling_model: str = CEILING_MODEL) -> Path:
    ceiling = next(
        (p["metrics"]["sensitivity_at_capacity"] for p in payloads if p["model"] == ceiling_model),
        None,
    )
    if not ceiling:
        return None

    rows = sorted(
        ((p["model"], 100 * p["metrics"]["sensitivity_at_capacity"] / ceiling) for p in payloads),
        key=lambda r: r[1],
    )
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = _colours(names)

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.6))
    bars = ax.barh(
        names,
        values,
        color=["black" if n == ceiling_model else colours[n] for n in names],
        height=0.6,
    )
    for bar, value in zip(bars, values):
        ax.text(
            value - 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="right",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.axvline(100, color="crimson", linewidth=1.0, linestyle="--")
    _style(ax, "Performance as a share of the achievable ceiling", "% of ceiling", "")
    ax.set_xlim(0, max(values) * 1.06)
    return _save(fig, path)


def shift_comparison(all_payloads: list[dict], path: Path) -> Path:
    by_split: dict[str, dict[str, dict]] = {}
    for payload in all_payloads:
        by_split.setdefault(payload["eval_split"], {})[payload["model"]] = payload["metrics"]

    if len(by_split) < 2:
        return None

    # Chronological, not alphabetical: shift_2020_11 would otherwise precede test_2020_04.
    canonical = list(config.SPLITS)
    splits = sorted(by_split, key=lambda s: canonical.index(s) if s in canonical else len(canonical))
    shared = sorted(set.intersection(*(set(by_split[s]) for s in splits)))
    if not shared:
        return None

    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("sensitivity_at_capacity", "Sensitivity @ capacity")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.4))
    positions = np.arange(len(shared))
    width = 0.8 / len(splits)

    for ax, (key, label) in zip(axes, metrics):
        for i, split in enumerate(splits):
            values = [by_split[split][m][key] for m in shared]
            ax.bar(positions + i * width, values, width=width, label=split)
        ax.set_xticks(positions + width * (len(splits) - 1) / 2)
        ax.set_xticklabels(shared, rotation=30, ha="right", fontsize=7)
        _style(ax, label, "", label)
        ax.set_ylim(0, 1)

    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out week vs eight months later", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def pr_and_roc(payloads: list[dict], path: Path, ceiling_model: str = CEILING_MODEL) -> Path:
    colours = _colours(p["model"] for p in payloads)
    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(11, 4.6))

    for payload in sorted(payloads, key=lambda p: p["model"]):
        name = payload["model"]
        curve = _curve(payload["score_table"])
        colour = "black" if name == ceiling_model else colours[name]
        style = "--" if name == ceiling_model else "-"
        label = f"{name} ({payload['metrics']['pr_auc']:.3f})"
        pr_ax.plot(curve["sensitivity"][1:], curve["precision"][1:], label=label, color=colour, linestyle=style, linewidth=1.4)
        roc_ax.plot(
            curve["fpr"],
            curve["sensitivity"],
            label=f"{name} ({payload['metrics']['roc_auc']:.3f})",
            color=colour,
            linestyle=style,
            linewidth=1.4,
        )

    prevalence = payloads[0]["metrics"]["prevalence"]
    pr_ax.axhline(prevalence, color="grey", linestyle=":", linewidth=0.9)
    pr_ax.annotate(f"prevalence {prevalence:.1%}", xy=(0.55, prevalence + 0.02), fontsize=8, color="grey")
    _style(pr_ax, "Precision–recall (PR-AUC)", "Recall / sensitivity", "Precision / PPV")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.02)
    pr_ax.legend(fontsize=7, loc="upper right")

    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.9)
    _style(roc_ax, "ROC (ROC-AUC)", "False positive rate", "True positive rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    return _save(fig, path)


def calibration(payloads: list[dict], path: Path, ceiling_model: str = CEILING_MODEL) -> Path:
    colours = _colours(p["model"] for p in payloads)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    limit = 0.0
    for payload in sorted(payloads, key=lambda p: p["model"]):
        name = payload["model"]
        predicted, observed, weight = _reliability(payload["score_table"])
        if predicted.size == 0:
            continue
        limit = max(limit, predicted.max(), observed.max())
        ax.plot(
            predicted,
            observed,
            marker="o",
            markersize=4,
            linewidth=1.3,
            color="black" if name == ceiling_model else colours[name],
            linestyle="--" if name == ceiling_model else "-",
            label=f"{name} (Brier {payload['metrics']['brier']:.4f})",
        )

    limit = min(1.0, limit * 1.08)
    ax.plot([0, limit], [0, limit], color="grey", linestyle=":", linewidth=1.0)
    ax.annotate("perfect calibration", xy=(limit * 0.55, limit * 0.6), fontsize=8, color="grey", rotation=38)
    _style(ax, "Calibration (equal-mass bins)", "Mean predicted probability", "Observed positive rate")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, path)


CHARTS = [
    (
        "sensitivity_capacity.png",
        "Sensitivity vs testing capacity",
        "The headline metric is a single point on this curve. Shows whether the ranking holds at every testing budget or only at the reported capacity.",
    ),
    (
        "ceiling_bars.png",
        "Share of the achievable ceiling",
        "The ceiling is the empirical positive rate per feature pattern - the best any model can do on these eight binary features.",
    ),
    (
        "shift_comparison.png",
        "Temporal generalisation",
        "The same models scored on the held-out week and eight months later. PR-AUC degrades far harder than ROC-AUC.",
    ),
    (
        "pr_roc.png",
        "Precision-recall and ROC",
        "At single-digit prevalence the PR curve is the more honest view; ROC is shown for comparability with the published figures.",
    ),
    (
        "calibration.png",
        "Calibration",
        "Predicted probability against observed positive rate, in bins holding equal numbers of people. Curves below the diagonal are over-predicting risk.",
    ),
]

_RENDERERS = {
    "sensitivity_capacity.png": lambda primary, everything, path, ceiling: sensitivity_vs_capacity(primary, path, ceiling),
    "ceiling_bars.png": lambda primary, everything, path, ceiling: ceiling_bars(primary, path, ceiling),
    "shift_comparison.png": lambda primary, everything, path, ceiling: shift_comparison(everything, path),
    "pr_roc.png": lambda primary, everything, path, ceiling: pr_and_roc(primary, path, ceiling),
    "calibration.png": lambda primary, everything, path, ceiling: calibration(primary, path, ceiling),
}


def render_all(
    primary: list[dict],
    everything: list[dict],
    charts_dir: Path,
    rel_prefix: str = "charts",
    ceiling_model: str = CEILING_MODEL,
) -> list[dict]:
    """Render every chart, skipping any that lack the data they need.

    `everything` must already be restricted to one track: a figure spanning tracks would
    overlay two different populations on the same axes.
    """
    rendered = []
    for filename, title, caption in CHARTS:
        result = _RENDERERS[filename](primary, everything, charts_dir / filename, ceiling_model)
        if result is not None:
            rendered.append(
                {"file": f"{rel_prefix}/{filename}", "title": title, "caption": caption}
            )
    return rendered
