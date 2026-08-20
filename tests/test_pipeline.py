import zipfile
from pathlib import Path

import numpy as np
import pytest

from covidbench import config, data, plots, registry
from covidbench.metrics import evaluate, score_table, sensitivity_at_capacity


def _data_available() -> bool:
    try:
        data.resolve_path("v006")
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(not _data_available(), reason="covidpred data not present")


@needs_data
def test_cohort_matches_published_sizes():
    for split in ("train_2020_03", "test_2020_04"):
        X, y = data.get_split(split, verify=True)
        assert len(X) == config.EXPECTED_ROWS[split]
        assert int(y.sum()) == config.EXPECTED_POSITIVES[split]


@needs_data
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


def test_registry_discovers_models():
    assert "ceiling_lookup" in registry.available()
    assert "released_lgbm_all" in registry.available()


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


def test_zipped_csv_ignores_macosx_entries(tmp_path):
    # Mirrors how covidpred ships the data: a zip that also contains __MACOSX cruft.
    archive = tmp_path / "sample.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sample.csv", "test_date,cough\n2020-03-22,1\n")
        zf.writestr("__MACOSX/._sample.csv", "resource fork")

    frame = data._read_csv(archive)
    assert list(frame.columns) == ["test_date", "cough"]
    assert len(frame) == 1
