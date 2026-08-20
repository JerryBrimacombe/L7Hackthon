"""Run a model and append one result file. Results are never edited, only added."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from . import config, data, registry
from .metrics import evaluate


def run_model(name: str, eval_split: str, capacity: float, verify: bool = True) -> dict:
    spec = registry.get(name)
    features = list(spec.features)

    X_train, y_train = data.get_split("train_2020_03", features, verify=verify)
    X_eval, y_eval = data.get_split(eval_split, features, verify=verify)

    model = spec.factory()
    started = time.perf_counter()
    if not getattr(model, "pretrained", False):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(X_eval)[:, 1]
    return {
        "model": name,
        "notes": spec.notes,
        "features": features,
        "train_split": "train_2020_03",
        "eval_split": eval_split,
        "pretrained": bool(getattr(model, "pretrained", False)),
        "fit_seconds": round(fit_seconds, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": evaluate(y_eval, probabilities, capacity),
    }


def write_result(result: dict) -> str:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "")
    path = config.RESULTS_DIR / f"{result['model']}__{result['eval_split']}__{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or all registered models.")
    parser.add_argument("--model", help="Model name; omit with --all")
    parser.add_argument("--all", action="store_true", help="Run every registered model")
    parser.add_argument("--eval-split", default=config.DEFAULT_EVAL_SPLIT, choices=list(config.SPLITS))
    parser.add_argument("--capacity", type=float, default=config.DEFAULT_CAPACITY)
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--no-verify", action="store_true", help="Skip cohort size assertions")
    args = parser.parse_args()

    if args.list:
        for name in registry.available():
            print(f"{name:26} {registry.REGISTRY[name].notes}")
        for module, reason in registry.UNAVAILABLE.items():
            print(f"[unavailable] {module}: {reason}")
        return

    names = registry.available() if args.all else [args.model]
    if not names or names == [None]:
        parser.error("provide --model NAME or --all")

    for name in names:
        result = run_model(name, args.eval_split, args.capacity, verify=not args.no_verify)
        metrics = result["metrics"]
        print(
            f"{name:26} sens@{metrics['capacity']:.0%}={metrics['sensitivity_at_capacity']:.3f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
            f"scores={metrics['distinct_scores']}"
        )
        write_result(result)


if __name__ == "__main__":
    main()
