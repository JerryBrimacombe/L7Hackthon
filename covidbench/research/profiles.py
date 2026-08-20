"""Dataset profiles for research-only experiments.

Canonical benchmark code in covidbench.data remains unchanged; this module lets us
explore alternative missing-value assumptions explicitly.
"""
from __future__ import annotations

import pandas as pd

from .. import config, data

SYMPTOMS = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]
DEMOGRAPHICS = ["age_60_and_above", "gender"]
POLICIES = ("paper", "drop_any", "impute_mode", "keep_unknown_binary")

# Extra columns for the variant that treats "unknown" as a state rather than a 0.
# Same names as the canonical inclusive track, so results stay comparable to it.
INDICATOR_FEATURES = config.INDICATOR_FEATURES

# The source CSV spells missing values as the literal string "None", and the loader
# reads with keep_default_na=False, so nothing is NaN until we convert it here.
MISSING_SENTINELS = ["None", ""]


def _load_labelled(version: str) -> pd.DataFrame:
    """Positive/negative rows only, with the sentinel strings turned into real NA."""
    raw = data.load_raw(version)
    frame = raw[raw["corona_result"].isin(["positive", "negative"])].copy()
    return frame.replace({sentinel: pd.NA for sentinel in MISSING_SENTINELS})


def _to_binary_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for source, target in zip(SYMPTOMS, config.FEATURES_ALL[:5]):
        out[target] = pd.to_numeric(frame[source], errors="coerce")
    out["Age_60_plus"] = frame["age_60_and_above"].eq("Yes").fillna(False).astype(int)
    out["Male"] = frame["gender"].eq("male").fillna(False).astype(int)
    out["Contact_with_confirmed"] = (
        frame["test_indication"].eq("Contact with confirmed").fillna(False).astype(int)
    )
    out[config.TARGET] = (frame["corona_result"] == "positive").astype(int)
    out["test_date"] = frame["test_date"]
    return out


def build_profile(
    version: str = "v006",
    missing_policy: str = "paper",
    with_indicators: bool = False,
) -> pd.DataFrame:
    """Build a research cohort under an explicit missing-value assumption.

    Policies:
    - paper: the canonical benchmark cohort (drop missing demographics, symptoms -> 0)
    - drop_any: drop rows missing any symptom or demographic
    - impute_mode: fill missing symptoms and demographics with the observed mode
    - keep_unknown_binary: retain rows with unknown demographics, encoding unknown as 0

    with_indicators adds explicit `*_unknown` columns. They are constant zero for the
    policies that remove or impute unknown rows, and only informative alongside
    keep_unknown_binary.
    """
    if missing_policy not in POLICIES:
        raise ValueError(
            f"Unknown missing_policy '{missing_policy}'. Use one of: {', '.join(POLICIES)}."
        )

    # Delegate rather than re-implement, so this variant can never drift from the benchmark.
    if missing_policy == "paper":
        cohort = data.build_cohort(version).copy()
        if with_indicators:
            for column in INDICATOR_FEATURES:
                cohort[column] = 0
        return cohort

    frame = _load_labelled(version)
    unknown_age = frame["age_60_and_above"].isna().to_numpy()
    unknown_gender = frame["gender"].isna().to_numpy()

    if missing_policy == "drop_any":
        keep = ~(frame[SYMPTOMS + DEMOGRAPHICS].isna().any(axis=1)).to_numpy()
        frame = frame[keep]
        unknown_age, unknown_gender = unknown_age[keep], unknown_gender[keep]
    elif missing_policy == "impute_mode":
        for column in SYMPTOMS + DEMOGRAPHICS:
            mode = frame[column].mode(dropna=True)
            if not mode.empty:
                frame[column] = frame[column].fillna(mode.iloc[0])

    encoded = _to_binary_features(frame)
    symptom_features = config.FEATURES_ALL[:5]
    encoded[symptom_features] = encoded[symptom_features].fillna(0).astype(int)

    if with_indicators:
        # Imputation deliberately erases the distinction, so the flags are zero there.
        blank = missing_policy == "impute_mode"
        encoded["Age_60_unknown"] = 0 if blank else unknown_age.astype(int)
        encoded["Gender_unknown"] = 0 if blank else unknown_gender.astype(int)

    return encoded.reset_index(drop=True)