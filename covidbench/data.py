"""Cohort construction. Single owner - every model must use these exact splits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import zipfile

import pandas as pd

from . import config

SYMPTOM_COLUMNS = {
    "cough": "Cough",
    "fever": "Fever",
    "sore_throat": "Sore_throat",
    "shortness_of_breath": "Shortness_of_breath",
    "head_ache": "Headache",
}


def resolve_path(version: str) -> Path:
    name = config.DATASETS[version]
    base = config.covidpred_root() / "data"
    # Extracted locally as a folder; still zipped after a fresh clone in CI.
    candidates = [base / name / name, base / f"{name}.zip", base / name]
    # Fall back to a copy checked into this repo, so a fresh clone works unconfigured.
    candidates += [config.REPO_ROOT / name, config.REPO_ROOT / "data" / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name} under {base} or {config.REPO_ROOT}. "
        "Set COVIDPRED_ROOT to your covidpred checkout."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        # The v0083 archive also contains __MACOSX resource forks, which makes
        # pandas refuse the zip outright, so pick the real CSV member ourselves.
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.startswith("__MACOSX/")
                and not Path(name).name.startswith("._")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one CSV in {path}, found {members}")
            with archive.open(members[0]) as handle:
                return pd.read_csv(handle, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@lru_cache(maxsize=4)
def load_raw(version: str) -> pd.DataFrame:
    return _read_csv(resolve_path(version))


@lru_cache(maxsize=4)
def build_cohort(version: str) -> pd.DataFrame:
    """Apply the paper's inclusion rules and encode the 8 binary predictors."""
    raw = load_raw(version)
    df = raw[raw["corona_result"].isin(["positive", "negative"])]
    df = df[df["age_60_and_above"].isin(["Yes", "No"]) & df["gender"].isin(["male", "female"])]

    out = pd.DataFrame(index=df.index)
    for src, dst in SYMPTOM_COLUMNS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out["Age_60_plus"] = (df["age_60_and_above"] == "Yes").astype(int)
    out["Male"] = (df["gender"] == "male").astype(int)
    # "Abroad" deliberately collapses into 0, matching the released model's encoding.
    out["Contact_with_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(int)
    out[config.TARGET] = (df["corona_result"] == "positive").astype(int)
    out["test_date"] = df["test_date"]

    # Unreported symptoms count as absent; dropping them instead loses 17 rows
    # and undershoots the paper's published cohort of 51,831.
    symptom_cols = list(SYMPTOM_COLUMNS.values())
    out[symptom_cols] = out[symptom_cols].fillna(0).astype(int)
    return out.reset_index(drop=True)


@lru_cache(maxsize=4)
def build_inclusive_cohort(version: str) -> pd.DataFrame:
    """Same encoding as build_cohort, but keeps rows with unknown age or gender.

    Unknown demographics stay 0 in the original columns and are flagged separately, so a
    model can tell "known to be under 60" apart from "not recorded".
    """
    raw = load_raw(version)
    df = raw[raw["corona_result"].isin(["positive", "negative"])]

    out = pd.DataFrame(index=df.index)
    for src, dst in SYMPTOM_COLUMNS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out["Age_60_plus"] = (df["age_60_and_above"] == "Yes").astype(int)
    out["Male"] = (df["gender"] == "male").astype(int)
    out["Contact_with_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(int)
    out["Age_60_unknown"] = (~df["age_60_and_above"].isin(["Yes", "No"])).astype(int)
    out["Gender_unknown"] = (~df["gender"].isin(["male", "female"])).astype(int)
    out[config.TARGET] = (df["corona_result"] == "positive").astype(int)
    out["test_date"] = df["test_date"]

    symptom_cols = list(SYMPTOM_COLUMNS.values())
    out[symptom_cols] = out[symptom_cols].fillna(0).astype(int)
    return out.reset_index(drop=True)


def _cohort_frame(version: str, cohort: str) -> pd.DataFrame:
    if cohort == config.COHORT_PAPER:
        return build_cohort(version)
    if cohort == config.COHORT_INCLUSIVE:
        return build_inclusive_cohort(version)
    raise ValueError(f"Unknown cohort '{cohort}'. Use one of: {config.COHORT_PAPER}, {config.COHORT_INCLUSIVE}.")


def get_split(
    split: str,
    features: list[str] | None = None,
    verify: bool = True,
    cohort: str = config.COHORT_PAPER,
):
    start, end, version = config.SPLITS[split]
    frame = _cohort_frame(version, cohort)
    part = frame[(frame["test_date"] >= start) & (frame["test_date"] <= end)]

    expected_rows = (
        config.EXPECTED_ROWS if cohort == config.COHORT_PAPER else config.EXPECTED_ROWS_INCLUSIVE
    )
    expected_positives = (
        config.EXPECTED_POSITIVES
        if cohort == config.COHORT_PAPER
        else config.EXPECTED_POSITIVES_INCLUSIVE
    )
    if verify and split in expected_rows:
        expected_n = expected_rows[split]
        expected_pos = expected_positives[split]
        actual_pos = int(part[config.TARGET].sum())
        if len(part) != expected_n or actual_pos != expected_pos:
            raise AssertionError(
                f"{split} ({cohort}): got {len(part)} rows / {actual_pos} positives, "
                f"expected {expected_n} / {expected_pos}. Preprocessing has drifted."
            )

    if features:
        cols = list(features)
    else:
        cols = (
            config.FEATURES_ALL if cohort == config.COHORT_PAPER else config.FEATURES_INCLUSIVE
        )
    return part[cols].reset_index(drop=True), part[config.TARGET].to_numpy()

