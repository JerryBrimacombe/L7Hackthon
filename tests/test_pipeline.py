import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covidbench import config, data, plots, registry
from covidbench.calibration import ScoreCalibrator
from covidbench.metrics import (
    bootstrap_confidence_intervals,
    confusion_matrix_at_capacity,
    evaluate,
    metrics_from_score_table,
    score_table,
    sensitivity_at_capacity,
)


def _data_available() -> bool:
    try:
        data.resolve_path("v006")
        return True
    except FileNotFoundError:
        return False


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


needs_data = pytest.mark.skipif(not _data_available(), reason="covidpred data not present")
# The registry skips models whose library is missing; the tests should agree.
needs_lightgbm = pytest.mark.skipif(not _module_available("lightgbm"), reason="lightgbm not installed")


@needs_data
def test_cohort_matches_published_sizes():
    for split in ("train_2020_03", "test_2020_04"):
        X, y = data.get_split(split, verify=True)
        assert len(X) == config.EXPECTED_ROWS[split]
        assert int(y.sum()) == config.EXPECTED_POSITIVES[split]


@needs_data
@needs_lightgbm
def test_released_model_truth_table_is_exhaustive():
    from covidbench.truth_table import truth_table

    table = truth_table("released_lgbm_all")
    assert len(table) == 256
    assert table["probability"].between(0, 1).all()


def test_sensitivity_handles_ties():
    # All scores identical: any capacity should catch exactly that fraction of positives.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.full(10, 0.5)
    assert sensitivity_at_capacity(y, p, 0.5) == pytest.approx(0.5)


def test_evaluate_reports_score_granularity():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.1, 0.9])
    result = evaluate(y, p, capacity=0.5)
    assert result["distinct_scores"] == 2
    assert result["sensitivity_at_capacity"] == pytest.approx(1.0)


def test_confusion_matrix_counts_reconcile_with_labels():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    matrix = confusion_matrix_at_capacity(score_table(y, p), capacity=0.3)
    assert matrix["selected"] == 3
    assert matrix["tp"] + matrix["fn"] == pytest.approx(matrix["total_pos"])
    assert matrix["tn"] + matrix["fp"] == pytest.approx(matrix["total_neg"])
    assert matrix["tp"] == pytest.approx(2.0)


def test_confusion_matrix_handles_capacity_ties():
    y = np.array([1, 0, 1, 0])
    p = np.full(4, 0.5)
    matrix = confusion_matrix_at_capacity(score_table(y, p), capacity=0.5)
    assert matrix["selected"] == 2
    assert matrix["tp"] == pytest.approx(1.0)
    assert matrix["fp"] == pytest.approx(1.0)


def test_sigmoid_calibration_preserves_ranking():
    p = np.array([0.05, 0.2, 0.4, 0.8, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    calibrated = ScoreCalibrator("sigmoid").fit(p, y).predict(p)
    assert np.all(np.diff(calibrated) > 0)


def test_isotonic_calibration_is_bounded():
    p = np.array([0.05, 0.2, 0.4, 0.8, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    calibrated = ScoreCalibrator("isotonic").fit(p, y).predict(np.array([0.1, 0.5, 0.9]))
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_registry_discovers_models():
    assert "ceiling_lookup" in registry.available()
    assert "released_lgbm_all" in registry.available()
    assert "neural_network" in registry.available()
    assert "random_forest" in registry.available()
    assert "decision_tree" in registry.available()
    assert "gaussian_nb" in registry.available()


def test_score_table_is_a_sufficient_statistic():
    rng = np.random.default_rng(0)
    p = rng.choice([0.1, 0.25, 0.4, 0.8], size=2000)
    y = (rng.random(2000) < p).astype(int)

    table = score_table(y, p)
    assert sum(r["n"] for r in table) == len(y)
    assert sum(r["pos"] for r in table) == int(y.sum())

    # The curves the charts draw must match the metrics computed from raw predictions.
    curve = plots._curve(table)
    reconstructed_auc = float(np.trapezoid(curve["sensitivity"], curve["fpr"]))
    assert reconstructed_auc == pytest.approx(evaluate(y, p)["roc_auc"], abs=1e-9)


def test_metrics_reconstruct_exactly_from_score_table():
    rng = np.random.default_rng(7)
    p = rng.choice([0.05, 0.1, 0.25, 0.4, 0.8], size=5000)
    y = (rng.random(5000) < p).astype(int)

    direct = evaluate(y, p, capacity=0.1)
    reconstructed = metrics_from_score_table(score_table(y, p), capacity=0.1)

    for key in ("roc_auc", "pr_auc", "sensitivity_at_capacity"):
        assert reconstructed[key] == pytest.approx(direct[key], abs=1e-12)


def test_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(11)
    p = rng.choice([0.05, 0.2, 0.6], size=4000)
    y = (rng.random(4000) < p).astype(int)

    table = score_table(y, p)
    direct = evaluate(y, p, capacity=0.1)
    intervals = bootstrap_confidence_intervals(table, capacity=0.1, n_boot=200)

    for key in ("roc_auc", "pr_auc", "sensitivity_at_capacity"):
        assert intervals[f"{key}_ci_low"] <= direct[key] <= intervals[f"{key}_ci_high"]


def test_charts_render(tmp_path):
    payloads = [
        {
            "model": name,
            "eval_split": "test_2020_04",
            "metrics": {
                "capacity": 0.1,
                "prevalence": 0.08,
                "roc_auc": 0.9,
                "pr_auc": 0.65,
                "brier": 0.07,
                "sensitivity_at_capacity": sens,
            },
            "score_table": [
                {"p": 0.8, "n": 100, "pos": 70},
                {"p": 0.3, "n": 400, "pos": 90},
                {"p": 0.05, "n": 1500, "pos": 40},
            ],
        }
        for name, sens in (("ceiling_lookup", 0.75), ("logreg", 0.72))
    ]

    rendered = plots.render_all(payloads, payloads, tmp_path)
    assert rendered
    for chart in rendered:
        written = tmp_path / Path(chart["file"]).name
        assert written.is_file() and written.stat().st_size > 1000


def test_charts_use_the_ceiling_of_the_requested_track(tmp_path):
    # ceiling_bars returns None when it cannot find the named ceiling, so this would
    # silently drop a chart if a track's own ceiling were ignored.
    payloads = [
        {
            "model": name,
            "eval_split": "test_2020_04",
            "metrics": {
                "capacity": 0.1,
                "prevalence": 0.08,
                "roc_auc": 0.9,
                "pr_auc": 0.65,
                "brier": 0.07,
                "sensitivity_at_capacity": sens,
            },
            "score_table": [
                {"p": 0.8, "n": 100, "pos": 70},
                {"p": 0.3, "n": 400, "pos": 90},
                {"p": 0.05, "n": 1500, "pos": 40},
            ],
        }
        for name, sens in (("ceiling_lookup_inclusive", 0.75), ("logreg", 0.72))
    ]

    rendered = plots.render_all(
        payloads,
        payloads,
        tmp_path / "inclusive",
        rel_prefix="charts/inclusive",
        ceiling_model="ceiling_lookup_inclusive",
    )
    files = [c["file"] for c in rendered]
    assert "charts/inclusive/ceiling_bars.png" in files
    for chart in rendered:
        assert chart["file"].startswith("charts/inclusive/")
        assert (tmp_path / "inclusive" / Path(chart["file"]).name).is_file()

    # The default ceiling is absent from these payloads, so that chart must be skipped.
    default_files = [c["file"] for c in plots.render_all(payloads, payloads, tmp_path / "default")]
    assert "charts/ceiling_bars.png" not in default_files


def test_zipped_csv_ignores_macosx_entries(tmp_path):
    # Mirrors how covidpred ships the data: a zip that also contains __MACOSX cruft.
    archive = tmp_path / "sample.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sample.csv", "test_date,cough\n2020-03-22,1\n")
        zf.writestr("__MACOSX/._sample.csv", "resource fork")

    frame = data._read_csv(archive)
    assert list(frame.columns) == ["test_date", "cough"]
    assert len(frame) == 1


@needs_data
def test_paper_profile_matches_canonical_cohort():
    from covidbench.research.profiles import build_profile

    pd.testing.assert_frame_equal(build_profile(missing_policy="paper"), data.build_cohort("v006"))


@needs_data
def test_inclusive_cohort_is_a_superset_of_the_published_one():
    for split in ("train_2020_03", "test_2020_04"):
        paper, _ = data.get_split(split, verify=True, cohort=config.COHORT_PAPER)
        inclusive, y = data.get_split(split, verify=True, cohort=config.COHORT_INCLUSIVE)
        assert len(inclusive) > len(paper)
        assert len(inclusive) == config.EXPECTED_ROWS_INCLUSIVE[split]
        assert int(y.sum()) == config.EXPECTED_POSITIVES_INCLUSIVE[split]
        assert list(inclusive.columns) == config.FEATURES_INCLUSIVE
        assert inclusive[config.FEATURES_INCLUSIVE].isin([0, 1]).all().all()


@needs_data
def test_indicator_flags_mark_exactly_the_rows_the_paper_drops():
    inclusive = data.build_inclusive_cohort("v006")
    paper = data.build_cohort("v006")
    known = inclusive[(inclusive["Age_60_unknown"] == 0) & (inclusive["Gender_unknown"] == 0)]
    assert len(known) == len(paper)


def test_tracks_declare_their_own_ceiling():
    # A shared ceiling across tracks would silently make different cohorts look comparable.
    ceilings = {meta["ceiling"] for meta in config.TRACKS.values()}
    assert len(ceilings) == len(config.TRACKS)
    for track, meta in config.TRACKS.items():
        spec = registry.get(meta["ceiling"])
        assert track in spec.tracks


def test_inclusive_track_hands_models_the_indicator_features():
    spec = registry.get("logreg")
    assert registry.features_for(spec, config.COHORT_PAPER) == config.FEATURES_ALL
    assert registry.features_for(spec, config.COHORT_INCLUSIVE) == config.FEATURES_INCLUSIVE


def test_released_artifacts_are_not_offered_on_the_inclusive_track():
    # They are fixed 8-feature dumps; scoring them on another cohort is not a replication.
    for name in ("released_lgbm_all", "released_lgbm_balanced", "ceiling_lookup"):
        assert config.COHORT_INCLUSIVE not in registry.get(name).tracks


@needs_data
def test_missing_policies_actually_differ():
    # The loader keeps blanks as empty strings, so a policy that forgets to
    # normalise them silently produces the canonical cohort instead of a variant.
    from covidbench.research.profiles import POLICIES, build_profile

    sizes = {policy: len(build_profile(missing_policy=policy)) for policy in POLICIES}
    assert sizes["drop_any"] < sizes["paper"]
    assert sizes["keep_unknown_binary"] > sizes["paper"]
    for policy in POLICIES:
        frame = build_profile(missing_policy=policy)
        assert frame[config.FEATURES_ALL].isin([0, 1]).all().all()
