"""Build a leaderboard from results/*.json and render it for GitHub Pages."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config, plots
from .metrics import bootstrap_confidence_intervals

CEILING_MODEL = "ceiling_lookup"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>covidpred benchmark</title>
<style>
 body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0 auto; max-width: 64rem;
        padding: 2rem 1rem 4rem; color: #1b1b1b; line-height: 1.5; }}
 h1 {{ margin-bottom: .25rem; }}
 h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #e5e5e5; padding-bottom: .3rem; }}
 h3 {{ margin-top: 1.5rem; margin-bottom: .5rem; }}
 .lede {{ color: #444; }}
 .kicker {{ text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700;
          color: #3b5bdb; margin-bottom: .5rem; }}
 .panel {{ background: #f8f9fb; border: 1px solid #e6e8ee; border-radius: 8px; padding: 1rem 1.1rem;
          margin: 1.25rem 0 1.75rem; }}
 .panel ul {{ margin: .55rem 0 0 1.2rem; padding: 0; }}
 .callout {{ background: #fff8e8; border-left: 4px solid #d6a331; padding: .9rem 1rem; margin: 1.25rem 0; }}
 .small-table {{ font-size: .92rem; }}
 table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .5rem .6rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ background: #f4f4f4; }}
 figure {{ margin: 1.5rem 0 2rem; }}
 figure img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
 figcaption {{ color: #555; font-size: .9rem; margin-top: .5rem; }}
 footer {{ margin-top: 3rem; color: #777; font-size: .85rem; }}
</style>
</head>
<body>
<h1>covidpred benchmark</h1>
<p class="kicker">Clinical screening question</p>
<p class="lede">Can a small set of symptom and basic patient data help prioritise who should be tested for COVID-19?</p>
<p class="lede">This benchmark answers that question directly by comparing simple symptom-based models against a strong empirical ceiling derived from the same eight binary features. Evaluation split: <strong>{split}</strong>. Track: <strong>{track}</strong> ({track_label}).</p>

<div class="panel">
  <strong>Why there are two tracks</strong>
  <ul>
    <li>The source data records age and gender as <code>None</code> for a large share of people. How you treat those rows changes <em>who is in the cohort at all</em>, not just how a feature is encoded.</li>
    <li><strong>paper</strong> drops them. That is what the published study did, so it is the only track that can claim replication - but it throws information away.</li>
    <li><strong>inclusive</strong> keeps them and adds explicit <code>Age_60_unknown</code> / <code>Gender_unknown</code> features, so a model can distinguish "recorded as under 60" from "not recorded".</li>
    <li>The two tracks score different people, so they each carry their own ceiling and are never ranked against one another.</li>
  </ul>
</div>

<div class="panel">
  <strong>Question and what we did</strong>
  <ul>
    <li>We asked whether a compact set of routine symptoms and demographic variables is sufficient to identify likely COVID-positive cases.</li>
    <li>We replicated the original study, reconstructed the benchmark pipeline, and compared a transparent baseline with tree-based and neural-network models on the same data.</li>
    <li>We then evaluated each model on screening performance, ranking quality, and calibration so the operational trade-offs are explicit.</li>
  </ul>
</div>

<h2>Baseline and model comparisons</h2>
<h3>Logistic regression as baseline</h3>
<p>Logistic regression is the natural baseline: it is simple, interpretable, and provides a useful reference point for how far a linear rule can go with the same symptom inputs.</p>

<h3>XGBoost</h3>
<p>XGBoost is the strongest nonlinear benchmark in this comparison. It captures interactions between symptoms and demographics that a linear model cannot represent, although the improvement is modest in a feature space this compact and highly structured.</p>

<h3>Random forest</h3>
<p>Random forests provide a second tree-based benchmark. They help test whether an ensemble approach offers a more robust ranking of risk than a single boosted model, particularly when performance differences are less clear-cut across subgroups.</p>

<h3>Neural network</h3>
<p>The neural network is a deliberately small multilayer perceptron using the same eight binary features. Scaling, regularisation, early stopping, and a fixed random seed keep the comparison reproducible and reduce the risk of overfitting this compact feature space.</p>

<h2>Model evaluation</h2>
<p>We assess each model across three dimensions: screening performance when testing capacity is constrained, ranking performance across thresholds, and probability calibration. For a triage task, these are complementary views of the same problem rather than interchangeable ones.</p>

<div class="panel">
  <strong>Classification performance and ROC curves</strong>
  <ul>
    <li>ROC curves show how well each model separates positives from negatives across all thresholds.</li>
    <li>Precision-recall curves are especially informative here because COVID-19 positives are a minority class.</li>
    <li>Calibration plots show whether predicted probabilities are trustworthy or systematically over- or under-estimated.</li>
  </ul>
</div>

<h2>Meaning behind the metrics</h2>
<ul>
  <li><strong>Sensitivity at capacity</strong>: the proportion of true positive cases identified when only a limited share of people can be tested.</li>
  <li><strong>ROC-AUC</strong>: overall discrimination across thresholds; useful for comparing ranking quality, but not the same as a decision rule.</li>
  <li><strong>PR-AUC</strong>: more relevant when positives are comparatively rare and false positives are costly.</li>
  <li><strong>Brier score</strong>: how closely predicted probabilities match observed outcomes; lower values are better.</li>
  <li><strong>% of ceiling</strong>: how close the model comes to the empirical best achievable benchmark derived from the observed pattern rates.</li>
</ul>

<h2>False positives and false negatives</h2>
<p>For a screening tool, the cost of errors matters as much as the headline accuracy. A <strong>false negative</strong> means a likely infection is missed and not prioritised for testing. A <strong>false positive</strong> means a lower-risk person is escalated earlier than necessary, which consumes limited testing capacity.</p>
<p>That is why sensitivity at a fixed testing budget is the central operating metric in this project: it reflects how many true cases are found when testing resources are constrained.</p>

{table}

<h3>Classification summary at the operating threshold</h3>
<p>These values describe the practical trade-off at the benchmark operating point: which cases are prioritised for testing, and what proportion of people are incorrectly flagged or missed.</p>
{summary_table}

<h3>Explainability summary</h3>
<p>Model-specific directional coefficients or impurity-based feature importance, when available. Coefficients keep sign (risk up/down); importance values are non-directional magnitudes.</p>
{explainability_table}

<h3>Column meanings</h3>
<ul>
  <li><strong>model</strong>: model name.</li>
  <li><strong>sensitivity_at_capacity</strong>: true-positive rate (recall) when only the selected fraction of people can be tested. Higher is better.</li>
  <li><strong>sens_ci</strong>: 95% percentile bootstrap interval for sensitivity at capacity. Overlapping intervals mean the models are not distinguishable on this data.</li>
  <li><strong>pct_of_ceiling</strong>: model sensitivity as a percentage of the empirical ceiling, where 100 means it reaches the lookup-table benchmark.</li>
  <li><strong>roc_auc</strong>: area under the ROC curve; measures ranking quality across all thresholds.</li>
  <li><strong>roc_auc_ci</strong>: 95% percentile bootstrap interval for ROC-AUC.</li>
  <li><strong>pr_auc</strong>: area under the precision-recall curve; often more informative when positives are rare.</li>
  <li><strong>brier</strong>: Brier score for probability calibration; lower is better.</li>
  <li><strong>distinct_scores</strong>: number of unique predicted scores. With eight binary inputs, there are at most 256 distinct score values.</li>
  <li><strong>precision</strong>: proportion of selected high-risk patients who are truly positive. Higher is better.</li>
  <li><strong>recall</strong>: proportion of true positives captured at the operating threshold. This is the same screening idea as sensitivity at capacity.</li>
  <li><strong>specificity</strong>: proportion of truly negative cases kept out of the tested group. Higher is better.</li>
  <li><strong>false_positive_rate</strong>: proportion of negatives incorrectly prioritised for testing. Lower is better.</li>
  <li><strong>false_negative_rate</strong>: proportion of true positives missed by the operating threshold. Lower is better.</li>
  <li><strong>f1_score</strong>: harmonic mean of precision and recall; a compact summary of the classification trade-off.</li>
</ul>

<div class="callout">
  <strong>Interpretation.</strong> A model that captures around 75% of true positives at a 10% testing budget is useful as a triage aid, even if it is not a definitive diagnostic test. The ranking metric (ROC-AUC) and the screening metric (sensitivity at capacity) tell slightly different stories, so both should be read together.
</div>

<div class="callout">
  <strong>Read the intervals, not the ordering.</strong> Several models sit within a fraction of a percentage point of each other. Where the bootstrap intervals overlap, the leaderboard ordering is not evidence that one model is better than another - it reflects sampling noise on a single evaluation week.
</div>

<div class="callout">
  <strong>Caveat.</strong> The lookup-table ceiling is the <em>in-sample</em> optimum, not a strict out-of-sample bound. Only a subset of patterns seen in the holdout week are present in training, so unseen or sparse patterns fall back to the overall prior. That is why a model can appear to slightly exceed 100% of ceiling in finite data.
</div>

{charts}

{tracks_section}

<footer>Generated {generated} &middot; {n_models} models &middot; built by <code>covidbench.compare</code></footer>
</body>
</html>
"""

FIGURE = """<figure>
<img src="{file}" alt="{title}">
<figcaption><strong>{title}.</strong> {caption}</figcaption>
</figure>
"""


def load_payloads(eval_split: str | None = None) -> list[dict]:
    """Latest result per (model, track, eval_split), optionally filtered to one split."""
    payloads: dict[tuple[str, str, str], dict] = {}
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if eval_split and payload["eval_split"] != eval_split:
            continue
        # Results written before tracks existed are all published-cohort runs.
        payload.setdefault("track", config.DEFAULT_TRACK)
        key = (payload["model"], payload["track"], payload["eval_split"])
        if key not in payloads or payload["timestamp"] >= payloads[key]["timestamp"]:
            payloads[key] = payload
    if not payloads:
        raise SystemExit("No results found. Run: python -m covidbench.run --all")
    return list(payloads.values())


def to_frame(payloads: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": p["model"],
                "track": p.get("track", config.DEFAULT_TRACK),
                "eval_split": p["eval_split"],
                "timestamp": p["timestamp"],
                **p["metrics"],
            }
            for p in payloads
        ]
    )


def leaderboard(frame: pd.DataFrame, track: str = config.DEFAULT_TRACK) -> pd.DataFrame:
    board = frame.sort_values("sensitivity_at_capacity", ascending=False).copy()
    ceiling_model = config.TRACKS[track]["ceiling"]
    ceiling = board.loc[board["model"] == ceiling_model, "sensitivity_at_capacity"]
    if not ceiling.empty and ceiling.iloc[0] > 0:
        board["pct_of_ceiling"] = 100 * board["sensitivity_at_capacity"] / ceiling.iloc[0]
    columns = ["model", "sensitivity_at_capacity", "sens_ci", "pct_of_ceiling", "roc_auc",
               "roc_auc_ci", "pr_auc", "brier", "distinct_scores"]
    return board[[c for c in columns if c in board.columns]]


def add_confidence_intervals(
    frame: pd.DataFrame, payloads: list[dict], n_boot: int
) -> pd.DataFrame:
    """Attach percentile bootstrap intervals so near-identical models are not over-read."""
    by_model = {p["model"]: p for p in payloads}
    sens_text: list[str] = []
    roc_text: list[str] = []

    for model in frame["model"]:
        payload = by_model.get(model, {})
        table = payload.get("score_table")
        if not table:
            sens_text.append("")
            roc_text.append("")
            continue
        capacity = float(payload["metrics"].get("capacity", config.DEFAULT_CAPACITY))
        intervals = bootstrap_confidence_intervals(table, capacity=capacity, n_boot=n_boot)
        sens_text.append(
            f"{intervals['sensitivity_at_capacity_ci_low']:.4f}-{intervals['sensitivity_at_capacity_ci_high']:.4f}"
            if "sensitivity_at_capacity_ci_low" in intervals
            else ""
        )
        roc_text.append(
            f"{intervals['roc_auc_ci_low']:.4f}-{intervals['roc_auc_ci_high']:.4f}"
            if "roc_auc_ci_low" in intervals
            else ""
        )

    frame = frame.copy()
    frame["sens_ci"] = sens_text
    frame["roc_auc_ci"] = roc_text
    return frame


def classification_summary(payloads: list[dict]) -> pd.DataFrame:
    """Compute operating-point summaries at the capacity-based screening threshold."""
    rows: list[dict] = []
    for payload in payloads:
        score_table = sorted(payload.get("score_table", []), key=lambda r: r["p"], reverse=True)
        if not score_table:
            continue

        total_n = sum(int(r["n"]) for r in score_table)
        total_pos = sum(int(r["pos"]) for r in score_table)
        total_neg = total_n - total_pos
        capacity = float(payload["metrics"].get("capacity", 0.1))
        target = max(1, int(round(capacity * total_n)))

        selected_n = 0
        tp = 0.0
        fp = 0.0
        for row in score_table:
            if selected_n >= target:
                break
            n = int(row["n"])
            pos = int(row["pos"])
            take = min(target - selected_n, n)
            tp += float(pos) * take / n if n else 0.0
            fp += float(n - pos) * take / n if n else 0.0
            selected_n += take

        fn = max(total_pos - tp, 0.0)
        tn = max(total_neg - fp, 0.0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        specificity = tn / total_neg if total_neg > 0 else 0.0
        fpr = fp / total_neg if total_neg > 0 else 0.0
        fnr = fn / total_pos if total_pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        threshold = score_table[min(len(score_table) - 1, max(0, selected_n - 1))]["p"]
        rows.append(
            {
                "model": payload["model"],
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "f1_score": f1,
                "operating_threshold": threshold,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("recall", ascending=False).reset_index(drop=True)


def explainability_summary(payloads: list[dict], top_k: int = 3) -> pd.DataFrame:
    rows: list[dict] = []
    for payload in sorted(payloads, key=lambda p: p["model"]):
        explainability = payload.get("explainability") or {}
        method = explainability.get("method")
        top_features = explainability.get("top_features") or []
        if not method or not top_features:
            rows.append({"model": payload["model"], "method": "n/a", "top_features": "n/a"})
            continue

        snippets: list[str] = []
        for feature in top_features[:top_k]:
            name = feature["feature"]
            value = feature.get("value")
            if value is None:
                snippets.append(name)
            else:
                snippets.append(f"{name} ({value:+.4f})")

        rows.append(
            {
                "model": payload["model"],
                "method": method,
                "top_features": ", ".join(snippets),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark results.")
    parser.add_argument("--eval-split", default=config.DEFAULT_EVAL_SPLIT)
    parser.add_argument(
        "--track",
        default=config.DEFAULT_TRACK,
        choices=list(config.TRACKS),
        help="Track whose leaderboard and charts are the headline ones",
    )
    parser.add_argument("--html", action="store_true", help="Also write docs/index.html")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart rendering")
    parser.add_argument("--no-ci", action="store_true", help="Skip bootstrap confidence intervals")
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap resamples per model")
    args = parser.parse_args()

    everything = load_payloads()
    primary = [
        p
        for p in everything
        if p["eval_split"] == args.eval_split and p.get("track", config.DEFAULT_TRACK) == args.track
    ]
    if not primary:
        raise SystemExit(f"No results for split '{args.eval_split}' on track '{args.track}'.")

    frame = to_frame(primary)
    if not args.no_ci:
        frame = add_confidence_intervals(frame, primary, args.n_boot)
    board = leaderboard(frame, args.track)
    summary = classification_summary(primary)
    explainability = explainability_summary(primary)

    print(f"=== track: {args.track} - {config.TRACKS[args.track]['label']} ===")
    print(board.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Other tracks are scored on a different population, so they get their own table.
    other_tracks = sorted(
        {
            p.get("track", config.DEFAULT_TRACK)
            for p in everything
            if p["eval_split"] == args.eval_split
        }
        - {args.track}
    )
    other_boards: list[tuple[str, pd.DataFrame]] = []
    for track in other_tracks:
        rows = [
            p
            for p in everything
            if p["eval_split"] == args.eval_split and p.get("track", config.DEFAULT_TRACK) == track
        ]
        other_frame = to_frame(rows)
        if not args.no_ci:
            other_frame = add_confidence_intervals(other_frame, rows, args.n_boot)
        other_board = leaderboard(other_frame, track)
        other_boards.append((track, other_board))
        print(f"\n=== track: {track} - {config.TRACKS[track]['label']} ===")
        print(other_board.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if other_boards:
        print(
            "\nTracks are scored on different cohorts. Compare models within a track;"
            " never read one track's numbers against another's."
        )

    if not args.html:
        return

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    charts_html = ""
    if not args.no_charts:
        plottable = [p for p in primary if p.get("score_table")]
        if plottable:
            rendered = plots.render_all(
                plottable,
                [p for p in everything if p.get("score_table")],
                config.DOCS_DIR / "charts",
            )
            charts_html = "<h2>Charts</h2>\n" + "\n".join(FIGURE.format(**c) for c in rendered)
            print(f"\nRendered {len(rendered)} charts")
        else:
            print("\nNo score_table in results; re-run covidbench.run to enable charts.")

    summary_html = summary.to_html(
        index=False,
        float_format=lambda v: f"{v:.4f}",
        border=0,
        classes="dataframe small-table",
    ) if not summary.empty else "<p>No classification summary available.</p>"

    explainability_html = explainability.to_html(
        index=False,
        border=0,
        classes="dataframe small-table",
    ) if not explainability.empty else "<p>No explainability summary available.</p>"

    tracks_html = ""
    for track, other_board in other_boards:
        meta = config.TRACKS[track]
        tracks_html += (
            f"<h3>{track} - {meta['label']}</h3>\n"
            f"<p>{meta['why']}</p>\n"
            + other_board.to_html(
                index=False, float_format=lambda v: f"{v:.4f}", border=0, classes="dataframe small-table"
            )
            + "\n"
        )
    if tracks_html:
        tracks_html = (
            "<h2>Other tracks</h2>\n"
            "<div class=\"callout\"><strong>Do not compare across tracks.</strong> Each track is scored "
            "on a different set of people, so the denominators differ. A higher number in one track "
            "does not mean a better model than a lower number in another - it usually means an easier "
            "or harder population. Each track has its own ceiling and its own <code>% of ceiling</code> "
            "column for exactly this reason.</div>\n" + tracks_html
        )

    html = PAGE.format(
        split=args.eval_split,
        track=args.track,
        track_label=config.TRACKS[args.track]["label"],
        tracks_section=tracks_html,
        table=board.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0),
        summary_table=summary_html,
        explainability_table=explainability_html,
        charts=charts_html,
        generated=max(p["timestamp"] for p in primary),
        n_models=len(primary),
    )
    (config.DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {config.DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
