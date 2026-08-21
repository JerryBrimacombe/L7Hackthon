"""Exhaustive truth table over the binary input space - the exact replication proof.

The all-features model has 8 binary inputs, so 256 rows fully characterise it.
Any reimplementation either reproduces these numbers or it does not.
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from . import config, data, registry


def truth_table(name: str, track: str = config.DEFAULT_TRACK) -> pd.DataFrame:
    spec = registry.get(name)
    if track not in spec.tracks:
        raise ValueError(f"Model '{name}' is not registered for track '{track}'.")

    features = registry.features_for(spec, track)
    grid = pd.DataFrame(
        list(itertools.product([0, 1], repeat=len(features))), columns=features
    )

    model = spec.factory()
    if not getattr(model, "pretrained", False):
        X_train, y_train = data.get_split(
            "train_2020_03", features, cohort=config.TRACKS[track]["cohort"]
        )
        model.fit(X_train, y_train)

    grid["probability"] = model.predict_proba(grid)[:, 1]
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump a model's full input-space truth table.")
    parser.add_argument("--model", default="released_lgbm_all")
    parser.add_argument("--track", default=config.DEFAULT_TRACK, choices=list(config.TRACKS))
    args = parser.parse_args()

    table = truth_table(args.model, track=args.track)
    out_dir = config.RESULTS_DIR / "truth_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.model}__{args.track}.csv"
    table.to_csv(path, index=False, float_format="%.12f")
    print(f"{len(table)} rows -> {path}")


if __name__ == "__main__":
    main()
