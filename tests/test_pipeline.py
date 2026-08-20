import zipfile

import numpy as np
import pytest

from covidbench import config, data, registry
from covidbench.metrics import evaluate, sensitivity_at_capacity


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


def test_zipped_csv_ignores_macosx_entries(tmp_path):
    # Mirrors how covidpred ships the data: a zip that also contains __MACOSX cruft.
    archive = tmp_path / "sample.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sample.csv", "test_date,cough\n2020-03-22,1\n")
        zf.writestr("__MACOSX/._sample.csv", "resource fork")

    frame = data._read_csv(archive)
    assert list(frame.columns) == ["test_date", "cough"]
    assert len(frame) == 1
