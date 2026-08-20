"""Build a leaderboard from results/*.json and render it for GitHub Pages."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config

CEILING_MODEL = "ceiling_lookup"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>covidpred benchmark</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; padding: 0 1rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .5rem .6rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ background: #f4f4f4; }}
 caption {{ text-align: left; padding-bottom: .75rem; color: #555; }}
</style>
</head>
<body>
<h1>covidpred benchmark</h1>
<p>Evaluation split: <strong>{split}</strong>. Headline metric is sensitivity at a fixed
testing capacity; <em>% of ceiling</em> compares each model to the empirical lookup table,
which is the best achievable score on these features.</p>
{table}
<p><small>Generated {generated}</small></p>
</body>
</html>
"""


def load_results(eval_split: str | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if eval_split and payload["eval_split"] != eval_split:
            continue
        rows.append({"model": payload["model"], "eval_split": payload["eval_split"],
                     "timestamp": payload["timestamp"], **payload["metrics"]})
    if not rows:
        raise SystemExit("No results found. Run: python -m covidbench.run --all")
    frame = pd.DataFrame(rows).sort_values("timestamp")
    return frame.drop_duplicates(subset=["model", "eval_split"], keep="last")


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
    args = parser.parse_args()

    frame = load_results(args.eval_split)
    board = leaderboard(frame)
    print(board.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.html:
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        html = PAGE.format(
            split=args.eval_split,
            table=board.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0),
            generated=frame["timestamp"].max(),
        )
        (config.DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
        print(f"\nWrote {config.DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
