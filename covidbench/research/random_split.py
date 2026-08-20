"""Research runner for random-split experiments.

Purpose: compare notebook-style random splits against the canonical temporal
benchmark without changing core pipeline behavior.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split

from .. import config, registry
from ..explainability import summarize as explainability_summary
from ..metrics import evaluate, score_table
from .profiles import build_profile


def run_model_random(
    name: str,
    version: str,
    missing_policy: str,
    test_size: float,
    capacity: float,
    seed: int,
) -> dict:
    spec = registry.get(name)
    features = list(spec.features)

    frame = build_profile(version=version, missing_policy=missing_policy)
    X = frame[features]
    y = frame[config.TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = spec.factory()
    if not getattr(model, "pretrained", False):
        model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "notes": spec.notes,
        "features": features,
        "dataset_version": version,
        "split_strategy": "random_stratified",
        "missing_policy": missing_policy,
        "seed": seed,
        "test_size": test_size,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": evaluate(y_test, probabilities, capacity),
        "score_table": score_table(y_test, probabilities),
        "explainability": explainability_summary(model, features),
    }


def write_result(result: dict) -> str:
    out_dir = config.RESULTS_DIR / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "")
    path = out_dir / (
        f"{result['model']}__{result['split_strategy']}__{result['missing_policy']}__{stamp}.json"
    )
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-split research experiments.")
    parser.add_argument("--model", help="Model name; omit with --all")
    parser.add_argument("--all", action="store_true", help="Run every registered model")
    parser.add_argument("--dataset", default="v006", choices=list(config.DATASETS))
    parser.add_argument(
        "--missing-policy",
        default="paper",
        choices=["paper", "drop_any", "impute_mode", "keep_unknown_binary"],
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--capacity", type=float, default=config.DEFAULT_CAPACITY)
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        for name in registry.available():
            print(f"{name:26} {registry.REGISTRY[name].notes}")
        return

    names = registry.available() if args.all else [args.model]
    if not names or names == [None]:
        parser.error("provide --model NAME or --all")

    for name in names:
        result = run_model_random(
            name=name,
            version=args.dataset,
            missing_policy=args.missing_policy,
            test_size=args.test_size,
            capacity=args.capacity,
            seed=args.seed,
        )
        metrics = result["metrics"]
        print(
            f"{name:26} {result['missing_policy']:20} sens@{metrics['capacity']:.0%}={metrics['sensitivity_at_capacity']:.3f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
        )
        write_result(result)


if __name__ == "__main__":
    main()