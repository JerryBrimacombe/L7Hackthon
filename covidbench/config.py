"""Shared constants. Single owner - changing anything here invalidates every result."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"


def covidpred_root() -> Path:
    env = os.environ.get("COVIDPRED_ROOT")
    return Path(env) if env else REPO_ROOT.parent / "covidpred"


DATASETS = {
    "v006": "corona_tested_individuals_ver_006.english.csv",
    "v0083": "corona_tested_individuals_ver_0083.english.csv",
}

# Order is load-bearing: it matches feature_names in lgbm_model_all_features.txt.
FEATURES_ALL = [
    "Cough",
    "Fever",
    "Sore_throat",
    "Shortness_of_breath",
    "Headache",
    "Age_60_plus",
    "Male",
    "Contact_with_confirmed",
]
FEATURES_BALANCED = ["Cough", "Fever", "Age_60_plus", "Male", "Contact_with_confirmed"]

# The published cohort drops every row with an unknown age or gender - roughly 46% of
# the file. The inclusive track keeps those rows and says "unknown" out loud instead of
# silently encoding it as 0.
INDICATOR_FEATURES = ["Age_60_unknown", "Gender_unknown"]
FEATURES_INCLUSIVE = FEATURES_ALL + INDICATOR_FEATURES

# The released dumps spell age as "Age_60+", which LightGBM rejects when training.
RELEASED_ALL_NAMES = [f.replace("Age_60_plus", "Age_60+") for f in FEATURES_ALL]
RELEASED_BALANCED_NAMES = [f.replace("Age_60_plus", "Age_60+") for f in FEATURES_BALANCED]

TARGET = "label"

# (start, end, dataset version) - inclusive dates.
SPLITS = {
    "train_2020_03": ("2020-03-22", "2020-03-31", "v006"),
    "test_2020_04": ("2020-04-01", "2020-04-07", "v006"),
    "shift_2020_11": ("2020-11-01", "2020-11-07", "v0083"),
}

# Verified against the paper's reported cohort sizes.
EXPECTED_ROWS = {"train_2020_03": 51831, "test_2020_04": 47401}
EXPECTED_POSITIVES = {"train_2020_03": 4769, "test_2020_04": 3624}

COHORT_PAPER = "paper"
COHORT_INCLUSIVE = "inclusive"

# Pinned the same way as the published cohort, so drift in the inclusive rules is caught too.
EXPECTED_ROWS_INCLUSIVE = {"train_2020_03": 53020, "test_2020_04": 48462}
EXPECTED_POSITIVES_INCLUSIVE = {"train_2020_03": 4795, "test_2020_04": 3641}

# A track is a (cohort, feature space, ceiling) triple. Results from different tracks are
# scored on different populations, so they are never ranked against each other and never
# share a "% of ceiling" column.
TRACKS = {
    COHORT_PAPER: {
        "cohort": COHORT_PAPER,
        "ceiling": "ceiling_lookup",
        "label": "Published cohort, 8 binary features",
        "why": (
            "Reproduces the paper exactly. Rows with unknown age or gender are dropped, "
            "so this is the only track that can claim replication."
        ),
    },
    COHORT_INCLUSIVE: {
        "cohort": COHORT_INCLUSIVE,
        "ceiling": "ceiling_lookup_inclusive",
        "label": "Inclusive cohort, 8 features plus 2 unknown indicators",
        "why": (
            "Keeps the ~46% of rows the paper discards and encodes 'unknown' as its own "
            "state rather than as 0. Better use of the data, but no longer a replication."
        ),
    },
}
DEFAULT_TRACK = COHORT_PAPER

DEFAULT_EVAL_SPLIT = "test_2020_04"
DEFAULT_CAPACITY = 0.10
RANDOM_SEED = 42
