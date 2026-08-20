"""Build a leaderboard from results/*.json and render it for GitHub Pages."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config, plots

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
 .lede {{ color: #444; }}
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
<p class="lede">Replication of <a href="https://www.nature.com/articles/s41746-020-00372-6">Zoabi et al. (2020)</a>
and a comparison of alternative models on the same data. Evaluation split: <strong>{split}</strong>.</p>

<h2>Leaderboard</h2>
<p>Ranked by sensitivity at a fixed testing capacity - the fraction of true cases caught when only
that share of people can be tested. <em>% of ceiling</em> compares each model against the empirical
per-pattern positive rate, the best achievable score on these eight binary features.</p>
{table}

<h3>Column meanings</h3>
<ul>
  <li><strong>model</strong>: model name.</li>
  <li><strong>sensitivity_at_capacity</strong>: true-positive rate (recall) when only the selected fraction of people can be tested. Higher is better.</li>
  <li><strong>pct_of_ceiling</strong>: model sensitivity as a percentage of the empirical ceiling, where 100 means it reaches the lookup-table benchmark.</li>
  <li><strong>roc_auc</strong>: area under the ROC curve; measures ranking quality across all thresholds.</li>
  <li><strong>pr_auc</strong>: area under the precision-recall curve; often more informative when positives are rare.</li>
  <li><strong>brier</strong>: Brier score for probability calibration; lower is better.</li>
  <li><strong>distinct_scores</strong>: number of unique predicted scores. With eight binary inputs, there are at most 256 distinct score values.</li>
</ul>

{charts}

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
    """Latest result per (model, eval_split), optionally filtered to one split."""
    payloads: dict[tuple[str, str], dict] = {}
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if eval_split and payload["eval_split"] != eval_split:
            continue
        key = (payload["model"], payload["eval_split"])
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
                "eval_split": p["eval_split"],
                "timestamp": p["timestamp"],
                **p["metrics"],
            }
            for p in payloads
        ]
    )


def leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    board = frame.sort_values("sensitivity_at_capacity", ascending=False).copy()
    ceiling = board.loc[board["model"] == CEILING_MODEL, "sensitivity_at_capacity"]
    if not ceiling.empty and ceiling.iloc[0] > 0:
        board["pct_of_ceiling"] = 100 * board["sensitivity_at_capacity"] / ceiling.iloc[0]
    columns = ["model", "sensitivity_at_capacity", "pct_of_ceiling", "roc_auc",
               "pr_auc", "brier", "distinct_scores"]
    return board[[c for c in columns if c in board.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark results.")
    parser.add_argument("--eval-split", default=config.DEFAULT_EVAL_SPLIT)
    parser.add_argument("--html", action="store_true", help="Also write docs/index.html")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart rendering")
    args = parser.parse_args()

    everything = load_payloads()
    primary = [p for p in everything if p["eval_split"] == args.eval_split]
    if not primary:
        raise SystemExit(f"No results for split '{args.eval_split}'.")

    board = leaderboard(to_frame(primary))
    print(board.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

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

    html = PAGE.format(
        split=args.eval_split,
        table=board.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0),
        charts=charts_html,
        generated=max(p["timestamp"] for p in primary),
        n_models=len(primary),
    )
    (config.DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {config.DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
