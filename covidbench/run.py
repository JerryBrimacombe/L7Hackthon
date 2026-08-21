"""Run a model and append one result file. Results are never edited, only added."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from . import config, data, registry
from .calibration import ScoreCalibrator
from .explainability import summarize as explainability_summary
from .metrics import evaluate, score_table


def run_model(
    name: str,
    eval_split: str,
    capacity: float,
    verify: bool = True,
    track: str = config.DEFAULT_TRACK,
    calibration_method: str | None = None,
) -> dict:
    spec = registry.get(name)
    if track not in spec.tracks:
        raise ValueError(f"Model '{name}' is not registered for track '{track}'.")

    cohort = config.TRACKS[track]["cohort"]
    features = registry.features_for(spec, track)

    train_split = "train_2020_03_model" if calibration_method else "train_2020_03"
    X_train, y_train = data.get_split(train_split, features, verify=verify, cohort=cohort)
    X_eval, y_eval = data.get_split(eval_split, features, verify=verify, cohort=cohort)

    model = spec.factory()
    started = time.perf_counter()
    if not getattr(model, "pretrained", False):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(X_eval)[:, 1]
    if calibration_method:
        X_cal, y_cal = data.get_split("calibration_2020_03", features, verify=verify, cohort=cohort)
        calibrator = ScoreCalibrator(calibration_method)
        calibrator.fit(model.predict_proba(X_cal)[:, 1], y_cal)
        probabilities = calibrator.predict(probabilities)
    result_model = name if not calibration_method else f"{name}__calibrated_{calibration_method}"
    explainability = explainability_summary(model, features)
    return {
        "model": result_model,
        "base_model": name,
        "notes": spec.notes,
        "features": features,
        "track": track,
        "cohort": cohort,
        "train_split": train_split,
        "calibration_split": "calibration_2020_03" if calibration_method else None,
        "calibration_method": calibration_method,
        "eval_split": eval_split,
        "pretrained": bool(getattr(model, "pretrained", False)),
        "fit_seconds": round(fit_seconds, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": evaluate(y_eval, probabilities, capacity),
        "score_table": score_table(y_eval, probabilities),
        "explainability": explainability,
    }


def write_result(result: dict) -> str:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "")
    path = (
        config.RESULTS_DIR
        / f"{result['model']}__{result['track']}__{result['eval_split']}__{stamp}.json"
    )
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or all registered models.")
    parser.add_argument("--model", help="Model name; omit with --all")
    parser.add_argument("--all", action="store_true", help="Run every registered model")
    parser.add_argument("--eval-split", default=config.DEFAULT_EVAL_SPLIT, choices=list(config.SPLITS))
    parser.add_argument(
        "--track",
        default=config.DEFAULT_TRACK,
        choices=list(config.TRACKS),
        help="paper = published cohort; inclusive = keeps unknown demographics",
    )
    parser.add_argument("--capacity", type=float, default=config.DEFAULT_CAPACITY)
    parser.add_argument(
        "--calibrate", choices=("sigmoid", "isotonic"),
        help="Fit a leakage-safe probability calibrator on the final March days.",
    )
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--no-verify", action="store_true", help="Skip cohort size assertions")
    args = parser.parse_args()

    if args.list:
        for name in registry.available():
            spec = registry.REGISTRY[name]
            print(f"{name:26} [{','.join(spec.tracks):17}] {spec.notes}")
        for module, reason in registry.UNAVAILABLE.items():
            print(f"[unavailable] {module}: {reason}")
        return

    names = registry.available_for(args.track) if args.all else [args.model]
    if not names or names == [None]:
        parser.error("provide --model NAME or --all")

    failures: list[str] = []
    for name in names:
        try:
            result = run_model(
                name, args.eval_split, args.capacity, verify=not args.no_verify, track=args.track,
                calibration_method=args.calibrate,
            )
        except Exception as exc:
            # A missing optional library must not cost everyone else their run.
            if not args.all:
                raise
            failures.append(name)
            print(f"{name:26} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        metrics = result["metrics"]
        print(
            f"{name:26} sens@{metrics['capacity']:.0%}={metrics['sensitivity_at_capacity']:.3f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
            f"scores={metrics['distinct_scores']}"
        )
        write_result(result)

    if failures:
        print(f"\n{len(failures)} model(s) skipped: {', '.join(failures)}")


if __name__ == "__main__":
    main()
