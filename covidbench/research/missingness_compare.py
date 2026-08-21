"""Batch comparison over missingness policies in the research track."""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config, registry
from .profiles import POLICIES
from .random_split import run_model_random, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare missingness policies for one or all models.")
    parser.add_argument("--model", help="Model name; omit with --all")
    parser.add_argument("--all", action="store_true", help="Run every registered model")
    parser.add_argument("--dataset", default="v006", choices=list(config.DATASETS))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--capacity", type=float, default=config.DEFAULT_CAPACITY)
    parser.add_argument(
        "--with-indicators",
        action="store_true",
        help="Add explicit Age_60_unknown / Gender_unknown features",
    )
    args = parser.parse_args()

    names = registry.available() if args.all else [args.model]
    if not names or names == [None]:
        parser.error("provide --model NAME or --all")

    rows = []
    for name in names:
        for policy in POLICIES:
            result = run_model_random(
                name=name,
                version=args.dataset,
                missing_policy=policy,
                test_size=args.test_size,
                capacity=args.capacity,
                seed=args.seed,
                with_indicators=args.with_indicators,
            )
            write_result(result)
            metrics = result["metrics"]
            rows.append(
                {
                    "model": name,
                    "missing_policy": policy,
                    "rows_scored": metrics["n"],
                    "prevalence": metrics["prevalence"],
                    "sensitivity_at_capacity": metrics["sensitivity_at_capacity"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "brier": metrics["brier"],
                }
            )

    frame = pd.DataFrame(rows)
    print(
        frame.sort_values(["model", "sensitivity_at_capacity"], ascending=[True, False]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    # Prevalence moves with the policy, so sensitivity at a fixed capacity is not
    # comparable across rows with different prevalence.
    print(
        "\nNote: cohort size and prevalence change with the policy, so compare within a"
        " policy first and treat cross-policy gaps as descriptive, not as a ranking."
    )


if __name__ == "__main__":
    main()