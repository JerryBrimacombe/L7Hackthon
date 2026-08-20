"""Dataset profiles for research-only experiments.

Canonical benchmark code in covidbench.data remains unchanged; this module lets us
explore alternative missing-value assumptions explicitly.
"""
from __future__ import annotations

import pandas as pd

from .. import config, data

SYMPTOMS = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]


def _to_binary_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["Cough"] = pd.to_numeric(frame["cough"], errors="coerce")
    out["Fever"] = pd.to_numeric(frame["fever"], errors="coerce")
    out["Sore_throat"] = pd.to_numeric(frame["sore_throat"], errors="coerce")
    out["Shortness_of_breath"] = pd.to_numeric(frame["shortness_of_breath"], errors="coerce")
    out["Headache"] = pd.to_numeric(frame["head_ache"], errors="coerce")
    out["Age_60_plus"] = (frame["age_60_and_above"] == "Yes").astype(float)
    out["Male"] = (frame["gender"] == "male").astype(float)
    out["Contact_with_confirmed"] = (
        frame["test_indication"] == "Contact with confirmed"
    ).astype(float)
    out["label"] = (frame["corona_result"] == "positive").astype(int)
    out["test_date"] = frame["test_date"]
    return out


def build_profile(version: str = "v006", missing_policy: str = "paper") -> pd.DataFrame:
    """Build a research cohort with an explicit missingness policy.

    Policies:
    - paper: canonical behavior (drop missing demographics, symptoms -> 0)
    - drop_any: drop rows with missing age/gender/symptoms
    - impute_mode: impute age/gender/symptoms with training-set mode proxies
    - keep_unknown_binary: keep missing demographics, map unknown to 0 side
    """
    raw = data.load_raw(version)
    frame = raw[raw["corona_result"].isin(["positive", "negative"])].copy()

    if missing_policy == "paper":
        frame = frame[
            frame["age_60_and_above"].isin(["Yes", "No"]) & frame["gender"].isin(["male", "female"])
        ].copy()
        encoded = _to_binary_features(frame)
        encoded[["Cough", "Fever", "Sore_throat", "Shortness_of_breath", "Headache"]] = (
            encoded[["Cough", "Fever", "Sore_throat", "Shortness_of_breath", "Headache"]]
            .fillna(0)
            .astype(int)
        )
        encoded[["Age_60_plus", "Male", "Contact_with_confirmed"]] = encoded[
            ["Age_60_plus", "Male", "Contact_with_confirmed"]
        ].astype(int)
        return encoded.reset_index(drop=True)

    if missing_policy == "drop_any":
        frame = frame.dropna(subset=SYMPTOMS + ["age_60_and_above", "gender"]).copy()
        frame = frame[
            frame["age_60_and_above"].isin(["Yes", "No"]) & frame["gender"].isin(["male", "female"])
        ].copy()
        encoded = _to_binary_features(frame)
        for col in config.FEATURES_ALL:
            encoded[col] = encoded[col].astype(int)
        return encoded.reset_index(drop=True)

    if missing_policy == "impute_mode":
        age_mode = frame["age_60_and_above"].mode(dropna=True)
        gender_mode = frame["gender"].mode(dropna=True)
        frame["age_60_and_above"] = frame["age_60_and_above"].fillna(age_mode.iloc[0] if not age_mode.empty else "No")
        frame["gender"] = frame["gender"].fillna(gender_mode.iloc[0] if not gender_mode.empty else "female")
        for symptom in SYMPTOMS:
            symptom_mode = frame[symptom].mode(dropna=True)
            fill = symptom_mode.iloc[0] if not symptom_mode.empty else "0"
            frame[symptom] = frame[symptom].fillna(fill)
        encoded = _to_binary_features(frame)
        for col in config.FEATURES_ALL:
            encoded[col] = encoded[col].fillna(0).astype(int)
        return encoded.reset_index(drop=True)

    if missing_policy == "keep_unknown_binary":
        frame["age_60_and_above"] = frame["age_60_and_above"].fillna("Unknown")
        frame["gender"] = frame["gender"].fillna("Unknown")
        encoded = _to_binary_features(frame)
        encoded[["Cough", "Fever", "Sore_throat", "Shortness_of_breath", "Headache"]] = (
            encoded[["Cough", "Fever", "Sore_throat", "Shortness_of_breath", "Headache"]]
            .fillna(0)
            .astype(int)
        )
        encoded[["Age_60_plus", "Male", "Contact_with_confirmed"]] = encoded[
            ["Age_60_plus", "Male", "Contact_with_confirmed"]
        ].astype(int)
        return encoded.reset_index(drop=True)

    raise ValueError(
        f"Unknown missing_policy '{missing_policy}'. "
        "Use one of: paper, drop_any, impute_mode, keep_unknown_binary."
    )