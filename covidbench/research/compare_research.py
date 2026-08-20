"""Aggregate research runs into a comparison table.

The canonical leaderboard in covidbench.compare deliberately ignores these files,
because research runs use different splits and cohorts and are not comparable to it.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .. import config


def load_payloads() -> list[dict]:
    research_dir = config.RESULTS_DIR / "research"
    payloads: dict[tuple, dict] = {}
    for path in sorted(research_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            payload["model"],
            payload.get("split_strategy"),
            payload.get("missing_policy"),
            payload.get("with_indicators", False),
            payload.get("dataset_version"),
        )
        if key not in payloads or payload["timestamp"] >= payloads[key]["timestamp"]:
            payloads[key] = payload
    if not payloads:
        raise SystemExit(
            "No research results found. Run: python -m covidbench.research.random_split --all"
        )
    return list(payloads.values())


def to_frame(payloads: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": p["model"],
                "dataset": p.get("dataset_version"),
                "missing_policy": p.get("missing_policy"),
                "indicators": p.get("with_indicators", False),
                "rows_scored": p["metrics"]["n"],
                "prevalence": p["metrics"]["prevalence"],
                "sensitivity_at_capacity": p["metrics"]["sensitivity_at_capacity"],
                "roc_auc": p["metrics"]["roc_auc"],
                "pr_auc": p["metrics"]["pr_auc"],
                "brier": p["metrics"]["brier"],
            }
            for p in payloads
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise research-track results.")
    parser.add_argument("--missing-policy", help="Filter to a single policy")
    parser.add_argument("--csv", action="store_true", help="Write results/research/summary.csv")
    args = parser.parse_args()

    frame = to_frame(load_payloads())
    if args.missing_policy:
        frame = frame[frame["missing_policy"] == args.missing_policy]
        if frame.empty:
            raise SystemExit(f"No research results for policy '{args.missing_policy}'.")

    frame = frame.sort_values(
        ["missing_policy", "sensitivity_at_capacity"], ascending=[True, False]
    ).reset_index(drop=True)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nPrevalence differs between policies, so sensitivity at a fixed capacity is"
        " only comparable within a policy."
    )

    if args.csv:
        out = config.RESULTS_DIR / "research" / "summary.csv"
        frame.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
