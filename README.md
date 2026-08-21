# L7 Hackathon â€” COVID Symptom Prediction Benchmark

A reproducible benchmark harness for the COVID-19 symptom-based prediction model published in
[*Machine learning-based prediction of COVID-19 diagnosis based on symptoms*](https://www.nature.com/articles/s41746-020-00372-6)
(Zoabi, Deri-Rozov & Shomron, npj Digital Medicine, 2020).

> **New here? Start with [Getting started](docs/GETTING_STARTED.md).** It explains how to install the
> project, find the data, run a first model, and open the generated leaderboard. You do not need to
> know GitHub or machine learning to follow it.

## Choose your path

| I want to... | Read this | Then run |
| --- | --- | --- |
| Get the benchmark running | [Getting started](docs/GETTING_STARTED.md) | `python -m covidbench.run --model logreg` |
| Understand the numbers | [Concepts and results](docs/CONCEPTS.md) | `python -m covidbench.compare --html` |
| Fix a setup or data problem | [Troubleshooting](docs/TROUBLESHOOTING.md) | Start with the matching error message |
| Add a model or improve the project | [Contributing](CONTRIBUTING.md) | `python -m pytest -q` |
| Browse the generated report | [Project guide](docs/README.md) | Open `docs/index.html` |

The examples in the detailed guide use the repository's virtual-environment Python executable. On
Windows use `.\.venv\Scripts\python.exe`; on macOS or Linux use `.venv/bin/python`.

This repo does two things:

1. **Replicates** the published LightGBM model from [nshomron/covidpred](https://github.com/nshomron/covidpred) and proves the replication is exact.
2. **Provides scaffolding** so the team can plug in alternative models and compare them against that baseline on identical data with identical metrics.

---

## Background

The original study predicts a positive COVID-19 test from eight binary inputs:

| Feature | Source column |
| --- | --- |
| `Cough` | `cough` |
| `Fever` | `fever` |
| `Sore_throat` | `sore_throat` |
| `Shortness_of_breath` | `shortness_of_breath` |
| `Headache` | `head_ache` |
| `Age_60_plus` | `age_60_and_above` |
| `Male` | `gender` |
| `Contact_with_confirmed` | `test_indication` |

The upstream repo ships **only model artifacts** â€” two LightGBM text dumps, a hyperparameter list, and the raw
data. There is no training code, no data pipeline, and no evaluation script. Everything here was
reconstructed from the paper and the serialised models.

### The eight-binary-feature consequence

Eight binary inputs means the entire input space is **2â¸ = 256 rows**. Two things follow, and they shape
the whole project:

- A model can be **completely characterised** by enumerating all 256 inputs. That is an exact replication
  test, not a statistical one.
- The empirical positive rate per pattern is the **Bayes-optimal predictor**. No model can beat it. This is
  registered as `ceiling_lookup`, and every other model is reported as a percentage of it.

The interesting question is therefore *not* "which model wins" â€” they all tie â€” but "how close to the
ceiling is everything, and is the extra complexity earning anything?"

---

## Key findings

### April 2020 holdout (the paper's evaluation week)

| Model | Sens @ 10% | % of ceiling | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `xgboost` | 0.7494 | 100.5 | 0.9028 | 0.7092 | 0.0346 |
| `lgbm_retrained` | 0.7462 | 100.0 | 0.9021 | 0.6968 | 0.0728 |
| `ceiling_lookup` | 0.7459 | 100.0 | 0.9023 | 0.7020 | 0.0349 |
| `released_lgbm_all` | 0.7311 | 98.0 | 0.8976 | 0.6499 | 0.0694 |
| `logreg` | 0.7255 | 97.3 | 0.8993 | 0.6675 | 0.0731 |
| `released_lgbm_balanced` | 0.6574 | 88.1 | 0.8630 | 0.4316 | 0.0864 |

- The released model scores **ROC-AUC 0.8976**, matching the published ~0.90. Replication confirmed.
- **Logistic regression reaches 97% of the theoretical ceiling.** The gradient boosting buys roughly half a
  point of sensitivity over a model you could print on a card.
- The spread from `xgboost` to `logreg` is about two points of sensitivity. Run
  `covidbench.compare` to see the bootstrap intervals: most of these models are **not**
  statistically separable on a single evaluation week.

> These tables were recorded before `random_forest`, `decision_tree`, `gaussian_nb` and
> `ceiling_lookup_balanced` were added, so they list fewer models than a current run. CI regenerates
> the live leaderboard on every push.

### November 2020 (temporal generalisation)

Same models, scored eight months later on the `v0083` dataset:

| Model | Sens @ 10% | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: |
| `xgboost` | 0.737 | 0.8543 | 0.5174 |
| `ceiling_lookup` | 0.737 | 0.8541 | 0.5077 |
| `lgbm_retrained` | 0.737 | 0.8516 | 0.5123 |
| `released_lgbm_all` | 0.737 | 0.8513 | 0.5053 |
| `logreg` | 0.727 | 0.8507 | 0.5062 |
| `released_lgbm_balanced` | 0.671 | 0.8125 | 0.3280 |

ROC-AUC falls ~5% (0.898 â†’ 0.851) while **PR-AUC falls ~22%** (0.650 â†’ 0.505). Judged on AUC alone the
degradation looks mild. It isn't â€” which is the metric argument in one line.

---

## Charts

`covidbench.compare --html` renders eight figures into `docs/charts/` and embeds them in the published
leaderboard.

| Chart | What it shows |
| --- | --- |
| Sensitivity vs capacity | The headline metric is one point on this curve; the detail panel zooms on the operating region |
| Share of ceiling | How much of the achievable maximum each model reaches |
| Temporal generalisation | April vs November, per metric â€” PR-AUC visibly degrades hardest |
| Precision-recall and ROC | Side by side, showing PR separates models that ROC makes look identical |
| Calibration | Equal-mass reliability curves against the diagonal |
| Calibration diagnostics | Calibration error, log loss, and calibration slope |
| Predicted probability distribution | Whether scores are compressed, extreme, or shifted from prevalence |
| Capacity-based confusion matrices | Raw counts and row percentages at the 10% prioritisation capacity |

### Charts are rendered per track

Each track gets its own chart set, drawn only from that track's results and highlighting that track's
ceiling in black:

- `docs/charts/` â€” the track chosen with `--track` (default `paper`)
- `docs/charts/<track>/` â€” every other track with results, shown under *Other tracks* on the page

This is not cosmetic. A single figure mixing tracks would overlay two different populations on the
same axes, and the share-of-ceiling bars would be measured against the wrong ceiling entirely.

The calibration chart earns its place: `xgboost` and `ceiling_lookup` sit on the diagonal, while
`lgbm_retrained`, `logreg` and `released_lgbm_balanced` fall well below it â€” they systematically
**over-predict risk**, a direct consequence of `is_unbalance=True` and `class_weight="balanced"`.
`released_lgbm_all` is near-vertical: with only 4 trees its predictions are compressed into roughly
0.13â€“0.21 while observed rates span 0.01â€“0.55. Good ranking, unusable probabilities.

The confusion-matrix chart uses the same operating point as the headline metric. â€œPrioritisedâ€ means
selected in the highest-ranked 10% for testing; â€œactual positiveâ€ means a later positive test. It is
therefore a capacity-based screening matrix, not the notebook's random-split matrix and not a diagnosis.

### Calibration experiments

Calibration is treated as a post-processing experiment, not as a replacement for the raw leaderboard.
The model is trained on 22â€“27 March, a calibrator is fitted on the disjoint 28â€“31 March calibration
period, and only then are probabilities scored on the untouched April or November evaluation week.
This prevents the evaluation labels from teaching the probability mapping.

Run a sigmoid calibrator, which is strictly monotonic and therefore preserves ranking:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all --calibrate sigmoid
.\.venv\Scripts\python.exe -m covidbench.run --all --calibrate sigmoid --eval-split shift_2020_11 --no-verify
.\.venv\Scripts\python.exe -m covidbench.compare --html --no-ci
```

Isotonic calibration is also available as a more flexible sensitivity analysis:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all --calibrate isotonic
```

Sigmoid is the recommended default for deployment-sized calibration sets. Isotonic can correct
non-linear distortions, but it creates stepwise probabilities and may introduce ties that slightly
change ROC-AUC and top-capacity ranking. Compare `brier`, `log_loss`, `calibration_error`, and
`calibration_slope` alongside ROC-AUC, PR-AUC, and sensitivity at capacity. Calibration should improve
probability interpretation; it is not expected to improve fixed-capacity ranking.

The November split is a temporal shift, not a calibration set. Its prevalence is materially lower than
April's, so a March-trained calibrator may still require a current-period prevalence adjustment before
being used operationally.

### How charts avoid storing predictions

Every figure is rebuilt from a `score_table` saved with each result â€” counts of people and positives per
distinct predicted score. Since eight binary features admit at most 256 distinct scores, this is a few KB
yet remains a **sufficient statistic**: ROC, PR, calibration and sensitivity at any capacity all reconstruct
from it exactly. A test asserts the reconstructed ROC-AUC matches the metric computed from raw predictions
to within 1e-9.

---

## Quick start

### Prerequisites

- **Python 3.13.** Not 3.14 â€” LightGBM has no wheels for it yet.
- A local checkout of [nshomron/covidpred](https://github.com/nshomron/covidpred) for the data.

By default the loader looks for `covidpred` as a **sibling directory**:

```text
parent/
  covidpred/      <- git clone https://github.com/nshomron/covidpred.git
  L7Hackthon/     <- this repo
```

Override with the `COVIDPRED_ROOT` environment variable if it lives elsewhere. Both zipped
(`*.csv.zip`, as cloned) and extracted layouts are handled automatically.

If neither is found, the loader falls back to a copy of the `v006` CSV checked into the repository
root, so a fresh clone runs the tests and the April benchmark with no setup at all. The `v0083`
dataset is not vendored, so the November shift split still needs a `covidpred` checkout.

Resolution order is: `COVIDPRED_ROOT/data/â€¦` â†’ sibling `covidpred/data/â€¦` â†’ repository root.

### Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-optional.txt
```

`requirements.txt` is the pinned core. `requirements-optional.txt` holds extra model libraries
(XGBoost, CatBoost) plus `seaborn`, which is only needed to re-run the archived notebooks. Models
importing a missing library are skipped rather than breaking everyone else's run.

### Run

```powershell
# Run the test suite
.\.venv\Scripts\python.exe -m pytest -q

# List registered models
.\.venv\Scripts\python.exe -m covidbench.run --list

# Benchmark everything on the paper's holdout week
.\.venv\Scripts\python.exe -m covidbench.run --all

# Run the small neural-network benchmark by itself
.\.venv\Scripts\python.exe -m covidbench.run --model neural_network

# Temporal shift test
.\.venv\Scripts\python.exe -m covidbench.run --all --eval-split shift_2020_11 --no-verify

# Leaderboard + GitHub Pages HTML
.\.venv\Scripts\python.exe -m covidbench.compare --html

# Leaderboard without bootstrap intervals (faster)
.\.venv\Scripts\python.exe -m covidbench.compare --no-ci

# 256-row replication proof
.\.venv\Scripts\python.exe -m covidbench.truth_table --model released_lgbm_all
```

### Reading the leaderboard

`compare` reports a 95% percentile bootstrap interval next to sensitivity at capacity and ROC-AUC.
Several models sit within a fraction of a point of one another, and with ~3,600 positives in the
evaluation week those gaps are mostly sampling noise. **Where intervals overlap, the ordering is not
evidence that one model beats another.**

The intervals are resampled from the stored `score_table`, so they can be recomputed for historical
results without re-running any model. Use `--n-boot` to trade precision for speed.

### Temporal split vs random split

- `Temporal split` (canonical benchmark): train on March, evaluate on April or November.
- `Random split` (research workflow): stratified random partitions drawn from one pool.

Why this matters: a random split draws train and test from the same weeks, so it leaks temporal
structure and inflates every metric. The canonical benchmark stays temporal because the most
interesting result in this repo â€” PR-AUC falling ~22% by November while ROC-AUC falls only ~5% â€” is
invisible under a random split.

Both are available. Use `covidbench.run` for results that belong on the leaderboard, and
`covidbench.research.*` for split and preprocessing experiments that deliberately do not.

---

## Research track

Everything under `covidbench/research/` is **deliberately excluded from the canonical leaderboard**.
It writes to `results/research/` and is aggregated by its own command. This is where notebook-derived
experiments live so they cannot silently contaminate the replication.

### Missing-value policies

The source CSV encodes missing values as the literal string `"None"`, and the loader reads with
`keep_default_na=False`, so **nothing is `NaN` until it is explicitly converted**. Any code that calls
`dropna()` or `fillna()` on the raw frame silently does nothing. `covidbench/research/profiles.py`
normalises the sentinel first.

This matters more than any hyperparameter in the repo: `age_60_and_above` is missing for **127,320 of
278,848 rows (~46%)** and `gender` for 19,563. Across the whole file the paper's rules discard about
half the data â€” though, as the two-tracks section explains, only ~2.2% inside the actual benchmark
windows. The canonical answer to this question is the `inclusive` track; the policies below are the
exploratory sweep that led to it.

| Policy | Rows (v006) | Prevalence | Rule |
| --- | ---: | ---: | --- |
| `paper` | 136,537 | 7.80% | Canonical cohort; delegates to `data.build_cohort` so it cannot drift |
| `drop_any` | 136,294 | 7.80% | Also drops the 243 rows with an unreported symptom |
| `impute_mode` | 274,956 | 5.36% | Fills missing symptoms and demographics with the observed mode |
| `keep_unknown_binary` | 274,956 | 5.36% | Keeps unknown-demographic rows, encoding unknown as 0 |

Two findings worth knowing before reading any comparison:

- **`impute_mode` and `keep_unknown_binary` are identical on v006.** The mode of `age_60_and_above`
  is `No` and of `gender` is `female`, both of which map to 0 â€” exactly what "unknown â†’ 0" does. They
  may diverge on `v0083`; that is untested.
- **Prevalence moves with the policy** (7.80% â†’ 5.36%). Sensitivity at a fixed 10% capacity is
  therefore *not* comparable across policies. Compare within a policy first.

### Missing indicators

`--with-indicators` adds explicit `Age_60_unknown` and `Gender_unknown` features instead of collapsing
unknown into 0. On `logreg` with `keep_unknown_binary` this is a real improvement:

| Encoding | Sens @ 10% | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: |
| unknown â†’ 0 | 0.7312 | 0.8835 | 0.5835 |
| explicit indicators | 0.7399 | 0.8913 | 0.5935 |

This experiment is what motivated the canonical `inclusive` track. Use the track for reportable
numbers; this flag remains for sweeping the idea across the other missing-value policies and across
random splits, which the canonical tracks deliberately do not do.

The flags are constant zero for `paper` and `impute_mode`, which remove or overwrite the unknowns.

### Research commands

```powershell
# Random stratified split
.\.venv\Scripts\python.exe -m covidbench.research.random_split --all --missing-policy paper

# One model, keeping unknown demographics as their own state
.\.venv\Scripts\python.exe -m covidbench.research.random_split --model logreg --missing-policy keep_unknown_binary --with-indicators

# Every policy side by side
.\.venv\Scripts\python.exe -m covidbench.research.missingness_compare --all

# Aggregate everything written to results/research/
.\.venv\Scripts\python.exe -m covidbench.research.compare_research --csv

# Maintained EDA artifact (Markdown + chart) into docs/research/
.\.venv\Scripts\python.exe -m covidbench.research.eda_report --dataset v006 --missing-policy paper
```

---

## The two tracks

This is the most important thing to understand before reading any number in this repo.

The source data records `age_60_and_above` and `gender` as the literal string `"None"` for a large
share of people. How you treat those rows decides **who is in the cohort at all** â€” it is a change of
population, not a change of feature encoding. There is no single right answer, so the repo runs both
and keeps them rigorously apart.

| | `paper` | `inclusive` |
| --- | --- | --- |
| Cohort | Drops rows with unknown age or gender | Keeps them |
| Features | 8 binary (or 5 for the balanced model) | 10 = the same 8 plus `Age_60_unknown`, `Gender_unknown` |
| Input space | 2â¸ = 256 | 2Â¹â° = 1024 |
| Ceiling | `ceiling_lookup` | `ceiling_lookup_inclusive` |
| Train / eval rows | 51,831 / 47,401 | 53,020 / 48,462 |
| Can claim replication? | **Yes** | No |

### Why both exist

**`paper` exists because replication is the point.** The published study drops these rows, so
reproducing its numbers requires dropping them too. This track is the only one allowed to say "we
replicated the paper". `data.build_cohort` is deliberately untouched by the inclusive work.

**`inclusive` exists because dropping data to encode it as 0 is lossy.** Under the paper's rules an
unknown age becomes `Age_60_plus = 0` â€” indistinguishable from someone confirmed to be under 60. The
inclusive track keeps the row and flags it, so the model can tell "recorded as under 60" apart from
"not recorded".

### Why they are never merged

Two independent reasons, and either alone would be enough:

1. **Different populations.** The tracks score different sets of people, so their denominators
   differ. A higher sensitivity in one track does not mean a better model â€” usually it means an
   easier or harder population.
2. **Different ceilings.** `ceiling_lookup` memorises the empirical positive rate per input pattern.
   Over 1024 patterns instead of 256 it memorises more noise, so the inclusive ceiling sits higher
   without anything actually being more achievable. Sharing a `pct_of_ceiling` column between the two
   would silently move the denominator under every historical result.

Enforcement is structural, not conventional: each entry in `config.TRACKS` names its own ceiling,
`compare` builds one leaderboard per track, and `tests/test_pipeline.py` asserts that no two tracks
share a ceiling model.

### Honest caveat: the benchmark weeks barely change

Missing demographics are **not** spread evenly over time. Reporting degraded sharply as testing
scaled up:

| Period | Rows | Unknown age or gender |
| --- | ---: | ---: |
| March 2020 (all) | 66,479 | 22.0% |
| April 2020 (all) | 208,477 | 59.4% |
| **`train_2020_03` window (22â€“31 Mar)** | **53,020** | **2.2%** |
| **`test_2020_04` window (1â€“7 Apr)** | **48,462** | **2.2%** |

The paper's evaluation windows sit in the period where recording was still good. So although the
`v006` file is ~50% unknown overall, the canonical splits lose only ~2.2% of rows â€” and the inclusive
track adds only about 1,100 rows to each split.

The measured consequence on `test_2020_04` is that the two tracks land within noise of each other:

| Model | `paper` sens@10% | `inclusive` sens@10% |
| --- | ---: | ---: |
| `random_forest` | 0.7494 | 0.7509 |
| `xgboost` | 0.7494 | 0.7508 |
| `logreg` | 0.7255 | 0.7245 |

**The inclusive track is not a free win on these splits.** Its value shows up when the evaluation
window includes the badly-recorded period â€” a random split across all of `v006` moves `logreg` from
0.7312 to 0.7399, because there half the rows have unknown demographics. Keep the inclusive track for
that reason and for later datasets like `v0083`, not because it flatters the April numbers.

### Running a track

```powershell
# Published cohort (default)
.\.venv\Scripts\python.exe -m covidbench.run --all

# Inclusive cohort
.\.venv\Scripts\python.exe -m covidbench.run --all --track inclusive

# Leaderboards: the chosen track leads, other tracks get their own table underneath
.\.venv\Scripts\python.exe -m covidbench.compare --html
.\.venv\Scripts\python.exe -m covidbench.compare --track inclusive

# 1024-row truth table for the inclusive input space
.\.venv\Scripts\python.exe -m covidbench.truth_table --model ceiling_lookup_inclusive --track inclusive
```

`--list` shows which tracks each model is registered for. A model opts in by declaring
`tracks=(config.COHORT_PAPER, config.COHORT_INCLUSIVE)` in its `@register`; the inclusive track then
hands it the two indicator columns automatically, so no model needs a duplicate file.

The released LightGBM artifacts and `ceiling_lookup` are **paper-only on purpose**. They are fixed
8-feature dumps whose whole value is exact replication, and a test asserts they are never offered on
the inclusive track.

---

```text
covidbench/
  config.py        Constants: paths, splits, feature order, track definitions. Single owner.
  data.py          Cohort rules (published and inclusive), splits, size assertions. Single owner.
  metrics.py       evaluate(), score_table(), bootstrap intervals - the one scoring path. Single owner.
  registry.py      Auto-discovers models/ and resolves features per track
  run.py           CLI; writes one JSON per run to results/
  compare.py       Builds one leaderboard per track and docs/index.html
  plots.py         Chart rendering, isolated from metrics and model files
  explainability.py Extracts coefficients or feature importances from a fitted model
  truth_table.py   Enumerates the full 2^n input space for a track
  models/
    released_lgbm.py    Published artifacts (baseline, paper track only)
    ceiling_lookup.py   Empirical per-pattern rates: 8-feature, 5-feature and inclusive ceilings
    lgbm_retrained.py   Retrained from published hyperparameters
    logreg.py           Interpretable linear benchmark
    random_forest.py    Notebook baseline, promoted
    decision_tree.py    Notebook baseline, promoted
    gaussian_nb.py      Notebook baseline, promoted
    neural_network.py   Small MLP benchmark with early stopping
    xgboost_clf.py      Peer GBM sanity check
  research/               Excluded from the canonical leaderboard by design
    profiles.py           Missing-value policy variants and optional unknown indicators
    random_split.py       Random stratified benchmark runner
    missingness_compare.py Runs every policy for one or all models
    compare_research.py   Aggregates results/research/*.json
    eda_report.py         Reproducible EDA artifacts into docs/research/
Notebooks/         Archived exploration, superseded by the modules above
tests/
  test_pipeline.py Cohort sizes, truth table, tie handling, zip loading, charts,
                   metric reconstruction, bootstrap coverage, missing-value policies,
                   track isolation and per-track ceilings
.github/workflows/
  bench.yml        CI: test, benchmark, publish to Pages
```

`results/` and `docs/` are generated and git-ignored; CI regenerates them.

### Notebooks

`Notebooks/` holds the original exploration. Its reusable content has already been extracted:
`RandomForest`, `DecisionTree` and `GaussianNB` are registered models, coefficient and
feature-importance summaries are in `explainability.py`, and the descriptive analysis is regenerated
by `research/eda_report.py`.

The notebooks are kept as a record of how the work was done, **not as a source of truth**. They use
their own random splits and preprocessing, so their numbers will not match the leaderboard. Their
data paths are relative to `Notebooks/`, and re-running them needs `seaborn` from
`requirements-optional.txt`.

---

## Adding a model

**Create one new file in `covidbench/models/`.** Nothing else. No shared file is edited, so five people can
work in parallel without merge conflicts.

```python
# covidbench/models/xgboost_clf.py
from xgboost import XGBClassifier

from .. import config
from ..registry import register

@register(
    "xgboost",
    features=config.FEATURES_ALL,
    notes="Peer GBM sanity check",
    tracks=(config.COHORT_PAPER, config.COHORT_INCLUSIVE),
)
def xgboost_clf():
    return XGBClassifier(max_depth=4, n_estimators=300, eval_metric="logloss")
```

`tracks` defaults to the `paper` track alone. Listing the inclusive track as well is all that is
needed to run on both cohorts â€” the registry appends the two indicator columns for you.

The contract is deliberately minimal â€” anything with **`fit(X, y)`** and **`predict_proba(X)`** works, which
means every scikit-learn, XGBoost, CatBoost and InterpretML estimator conforms already. There is no base
class to inherit.

Set `pretrained = True` on the returned object to skip fitting (used by the released artifacts).

If the estimator exposes `coef_` or `feature_importances_`, `covidbench.run` captures a compact
explainability summary into the result JSON and `covidbench.compare --html` renders it. Pipelines are
unwrapped automatically, so `make_pipeline(StandardScaler(), LogisticRegression())` still reports
coefficients. Coefficients keep their sign; importances are unsigned magnitudes. Models exposing
neither (for example `ceiling_lookup`) report `n/a`.

### The two ceilings

`ceiling_lookup` is the empirical positive rate per pattern over the 8 features, and is the bound for
every 8-feature model on the `paper` track. `released_lgbm_balanced` only sees 5 features, so that
bound does not apply to it â€” `ceiling_lookup_balanced` is its correct reference. The `pct_of_ceiling`
column is computed against the 8-feature ceiling, so read the balanced model against its own ceiling
row rather than the percentage.

The `inclusive` track has a third ceiling, `ceiling_lookup_inclusive`, over 1024 patterns. Every
ceiling is an **in-sample** optimum, and the larger the pattern space the more optimistic it gets:
held-out measurement puts the 8-feature ceiling about +0.0065 above its true value and the 10-feature
one about +0.0123. That is why a model can post slightly over 100% of ceiling, and why the two
tracks must never share the column.

Then:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --model xgboost
.\.venv\Scripts\python.exe -m covidbench.compare
```

### Rules that keep results comparable

- **Never** change `config.py`, `data.py` or `metrics.py` without telling the team. Every result becomes
  incomparable if the splits or scoring shift mid-hackathon.
- Results are **append-only**: one JSON file per run, never edited. Two people finishing at once produce two
  files, not a conflict.
- Everything trains on `train_2020_03` and is scored on the same evaluation split, within one track.
- **Never compare numbers across tracks.** They are different populations with different ceilings.
- Experiments that break any of the above belong in `covidbench/research/`, which writes to a
  separate directory and is never merged into the leaderboard.

---

## Data and cohort definition

Source: Israeli Ministry of Health public testing data, via `covidpred`.

- `v006` â€” downloaded 4 May 2020. **278,848 rows**, 11 Mar â€“ 30 Apr 2020. Used for replication.
- `v0083` â€” downloaded 15 Nov 2020. Used for the temporal shift test.

Raw `v006` label distribution: 260,227 negative / 14,729 positive / 3,892 `other` â€” roughly **5.3% prevalence**.

### Inclusion rules

1. Keep `corona_result` of `positive` or `negative` (drop `other`).
2. Drop rows with missing `age_60_and_above` or `gender`.
3. Encode unreported symptoms as absent (0).
4. `Contact_with_confirmed` is 1 only for `test_indication == "Contact with confirmed"`; `Abroad` collapses to 0.

### How missingness is encoded

The CSV writes missing values as the literal string `"None"`, not as an empty cell, and `data.py`
reads with `keep_default_na=False`. The canonical pipeline handles this by construction â€”
`pd.to_numeric(..., errors="coerce")` turns `"None"` into `NaN` for symptoms, and the `isin(["Yes",
"No"])` filter excludes it for demographics.

The trap is that `df.dropna()` and `df.fillna()` on the raw frame are **silent no-ops**. Anything
reading the raw CSV directly must normalise the sentinel first; see
`research/profiles.MISSING_SENTINELS`.

This reproduces the published cohort **exactly**, and [`tests/test_pipeline.py`](tests/test_pipeline.py) asserts it:

| Split | Window | Rows | Positives |
| --- | --- | ---: | ---: |
| `train_2020_03` | 22â€“31 Mar 2020 | 51,831 | 4,769 |
| `test_2020_04` | 1â€“7 Apr 2020 | 47,401 | 3,624 |
| `shift_2020_11` | 1â€“7 Nov 2020 | â€” | â€” |

If preprocessing ever drifts, those assertions fail loudly rather than silently corrupting every comparison.

---

## Metrics

`evaluate()` returns ROC-AUC, PR-AUC, Brier score, prevalence, distinct score count, and the headline metric.

### Headline: sensitivity at fixed testing capacity

> If we can only test 10% of people, what fraction of true cases do we catch?

This is preferred over ROC-AUC because:

1. **It matches the actual decision.** The paper's stated purpose is prioritising scarce tests â€” a
   budget-constrained ranking problem, not a threshold-free one.
2. **AUC averages over operating points nobody uses.** It weights the 90%-capacity regime equally with the 5%
   regime. Nobody deploys at 90%.
3. **It is directly interpretable.** "Test 10% of arrivals, catch 73% of cases" is actionable. "AUC 0.8976" is not.
4. **AUC hides prevalence shift**, as the November results above demonstrate.
5. **It exposes ranking ties.** With â‰¤256 distinct scores, huge groups of people share one score. AUC smooths
   this away; `sensitivity_at_capacity` computes the expected value under random tie-breaking, so results
   don't silently depend on row ordering.

ROC-AUC is still reported for comparability with the paper â€” just don't lead with it.

---

## Replication notes and gotchas

Three issues were found while reproducing the published model. All are handled in code; they are documented
here because each one fails silently or confusingly.

**1. Windows line endings corrupt the released model.**
LightGBM splits trees using byte offsets in the `tree_sizes` header. A checkout with `core.autocrlf=true`
rewrites the file with CRLF, invalidating every offset â€” LightGBM then **aborts the interpreter** with
`Model format error, expect a tree here`. `released_lgbm.py` normalises line endings in memory before loading.

**2. Missing symptom values.**
Dropping rows with unreported symptoms yields 51,814 training rows, not 51,831. Treating them as *absent*
reproduces the published cohort exactly.

**3. `__MACOSX` entries in the data archives.**
`covidpred` ships the CSVs zipped, and the `v0083` archive contains macOS resource forks. pandas refuses any
zip with multiple members, so `data.py` selects the real CSV member explicitly. This only manifests in CI,
where the data is still zipped.

### Known caveat: the ceiling is not a strict bound

`lgbm_retrained` scored **100.04%** of ceiling. The lookup table is the *in-sample* optimum: only ~74 of the
200 patterns present in the April week appear in March training data, so unseen patterns fall back to the
prior. Smoothing the table is an open task.

---

## Continuous integration

[`.github/workflows/bench.yml`](.github/workflows/bench.yml) runs on every branch push, on pull requests, and
on manual dispatch. It clones `covidpred` for data, runs the tests, benchmarks all models on both evaluation
windows, and uploads `results/` as an artifact.

| Event | Benchmark | Pages deploy |
| --- | --- | --- |
| Push to feature branch | Yes | No |
| Open / update PR | Yes (deduplicated) | No |
| Merge to `main` | Yes | Yes |
| Markdown-only commit | No | No |

The concurrency key resolves to the branch name for both `push` and `pull_request`, so a branch with an open
PR doesn't burn two runners on identical work.

**Before the first deploy:** enable Pages under *Settings â†’ Pages â†’ Source: GitHub Actions*. Without it, the
deploy job fails on `main` even when the benchmark passes.

---

## Roadmap

- [x] XGBoost peer comparison
- [x] Calibration curves alongside Brier scores
- [ ] Additional models: CatBoost, shallow decision tree, naive Bayes, Explainable Boosting Machine
- [ ] Subgroup breakdowns by age and gender
- [ ] Smoothing for the ceiling lookup table
- [ ] Symptom-count scorecard as a clinical baseline
- [ ] Sensitivity analysis: `NaN` symptoms (as the paper did) rather than imputing 0

Neural networks are **deliberately excluded**: with 256 possible inputs an MLP cannot outperform the lookup
table, so including one adds cost and invites "why didn't you tune it harder?"

---

## Interpreting any of this

The underlying model predicts **who tests positive**, not who is infected. It is trained on people who were
selected for testing, under a specific testing policy, in one country, in early 2020, pre-vaccination and
pre-variants. It is a triage-prioritisation aid, not a diagnostic.

## License

MIT.
