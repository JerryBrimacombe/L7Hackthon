# L7 Hackathon — COVID Symptom Prediction Benchmark

A reproducible benchmark harness for the COVID-19 symptom-based prediction model published in
[*Machine learning-based prediction of COVID-19 diagnosis based on symptoms*](https://www.nature.com/articles/s41746-020-00372-6)
(Zoabi, Deri-Rozov & Shomron, npj Digital Medicine, 2020).

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

The upstream repo ships **only model artifacts** — two LightGBM text dumps, a hyperparameter list, and the raw
data. There is no training code, no data pipeline, and no evaluation script. Everything here was
reconstructed from the paper and the serialised models.

### The eight-binary-feature consequence

Eight binary inputs means the entire input space is **2⁸ = 256 rows**. Two things follow, and they shape
the whole project:

- A model can be **completely characterised** by enumerating all 256 inputs. That is an exact replication
  test, not a statistical one.
- The empirical positive rate per pattern is the **Bayes-optimal predictor**. No model can beat it. This is
  registered as `ceiling_lookup`, and every other model is reported as a percentage of it.

The interesting question is therefore *not* "which model wins" — they all tie — but "how close to the
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

ROC-AUC falls ~5% (0.898 → 0.851) while **PR-AUC falls ~22%** (0.650 → 0.505). Judged on AUC alone the
degradation looks mild. It isn't — which is the metric argument in one line.

---

## Charts

`covidbench.compare --html` renders five figures into `docs/charts/` and embeds them in the published
leaderboard.

| Chart | What it shows |
| --- | --- |
| Sensitivity vs capacity | The headline metric is one point on this curve; the detail panel zooms on the operating region |
| Share of ceiling | How much of the achievable maximum each model reaches |
| Temporal generalisation | April vs November, per metric — PR-AUC visibly degrades hardest |
| Precision-recall and ROC | Side by side, showing PR separates models that ROC makes look identical |
| Calibration | Equal-mass reliability curves against the diagonal |

The calibration chart earns its place: `xgboost` and `ceiling_lookup` sit on the diagonal, while
`lgbm_retrained`, `logreg` and `released_lgbm_balanced` fall well below it — they systematically
**over-predict risk**, a direct consequence of `is_unbalance=True` and `class_weight="balanced"`.
`released_lgbm_all` is near-vertical: with only 4 trees its predictions are compressed into roughly
0.13–0.21 while observed rates span 0.01–0.55. Good ranking, unusable probabilities.

### How charts avoid storing predictions

Every figure is rebuilt from a `score_table` saved with each result — counts of people and positives per
distinct predicted score. Since eight binary features admit at most 256 distinct scores, this is a few KB
yet remains a **sufficient statistic**: ROC, PR, calibration and sensitivity at any capacity all reconstruct
from it exactly. A test asserts the reconstructed ROC-AUC matches the metric computed from raw predictions
to within 1e-9.

---

## Quick start

### Prerequisites

- **Python 3.13.** Not 3.14 — LightGBM has no wheels for it yet.
- A local checkout of [nshomron/covidpred](https://github.com/nshomron/covidpred) for the data.

By default the loader looks for `covidpred` as a **sibling directory**:

```
parent/
  covidpred/      <- git clone https://github.com/nshomron/covidpred.git
  L7Hackthon/     <- this repo
```

Override with the `COVIDPRED_ROOT` environment variable if it lives elsewhere. Both zipped
(`*.csv.zip`, as cloned) and extracted layouts are handled automatically.

### Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-optional.txt
```

`requirements.txt` is the pinned core. `requirements-optional.txt` holds extra model libraries
(XGBoost, CatBoost); models importing a missing library are skipped rather than breaking everyone else's run.

### Run

```powershell
# Verify the replication (6 tests)
.\.venv\Scripts\python.exe -m pytest -q

# List registered models
.\.venv\Scripts\python.exe -m covidbench.run --list

# Benchmark everything on the paper's holdout week
.\.venv\Scripts\python.exe -m covidbench.run --all

# Temporal shift test
.\.venv\Scripts\python.exe -m covidbench.run --all --eval-split shift_2020_11 --no-verify

# Leaderboard + GitHub Pages HTML
.\.venv\Scripts\python.exe -m covidbench.compare --html

# 256-row replication proof
.\.venv\Scripts\python.exe -m covidbench.truth_table --model released_lgbm_all
```

---

## Repository layout

```
covidbench/
  config.py        Constants: paths, splits, feature order. Single owner.
  data.py          Cohort rules, split construction, size assertions. Single owner.
  metrics.py       evaluate() and score_table() - the one scoring path. Single owner.
  registry.py      Auto-discovers everything in models/
  run.py           CLI; writes one JSON per run to results/
  compare.py       Builds the leaderboard and docs/index.html
  plots.py         Chart rendering, isolated from metrics and model files
  truth_table.py   Enumerates the full 2^n input space
  models/
    released_lgbm.py    Published artifacts (baseline)
    ceiling_lookup.py   Empirical per-pattern rate (upper bound)
    lgbm_retrained.py   Retrained from published hyperparameters
    logreg.py           Interpretable linear benchmark
    xgboost_clf.py      Peer GBM sanity check
tests/
  test_pipeline.py Cohort sizes, truth table, tie handling, zip loading, chart rendering
.github/workflows/
  bench.yml        CI: test, benchmark, publish to Pages
```

`results/` and `docs/` are generated and git-ignored; CI regenerates them.

---

## Adding a model

**Create one new file in `covidbench/models/`.** Nothing else. No shared file is edited, so five people can
work in parallel without merge conflicts.

```python
# covidbench/models/xgboost_clf.py
from xgboost import XGBClassifier

from .. import config
from ..registry import register


@register("xgboost", features=config.FEATURES_ALL, notes="Peer GBM sanity check")
def xgboost_clf():
    return XGBClassifier(max_depth=4, n_estimators=300, eval_metric="logloss")
```

The contract is deliberately minimal — anything with **`fit(X, y)`** and **`predict_proba(X)`** works, which
means every scikit-learn, XGBoost, CatBoost and InterpretML estimator conforms already. There is no base
class to inherit.

Set `pretrained = True` on the returned object to skip fitting (used by the released artifacts).

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
- Everything trains on `train_2020_03` and is scored on the same evaluation split.

---

## Data and cohort definition

Source: Israeli Ministry of Health public testing data, via `covidpred`.

- `v006` — downloaded 4 May 2020. **278,848 rows**, 11 Mar – 30 Apr 2020. Used for replication.
- `v0083` — downloaded 15 Nov 2020. Used for the temporal shift test.

Raw `v006` label distribution: 260,227 negative / 14,729 positive / 3,892 `other` — roughly **5.3% prevalence**.

### Inclusion rules

1. Keep `corona_result` of `positive` or `negative` (drop `other`).
2. Drop rows with missing `age_60_and_above` or `gender`.
3. Encode unreported symptoms as absent (0).
4. `Contact_with_confirmed` is 1 only for `test_indication == "Contact with confirmed"`; `Abroad` collapses to 0.

This reproduces the published cohort **exactly**, and [`tests/test_pipeline.py`](tests/test_pipeline.py) asserts it:

| Split | Window | Rows | Positives |
| --- | --- | ---: | ---: |
| `train_2020_03` | 22–31 Mar 2020 | 51,831 | 4,769 |
| `test_2020_04` | 1–7 Apr 2020 | 47,401 | 3,624 |
| `shift_2020_11` | 1–7 Nov 2020 | — | — |

If preprocessing ever drifts, those assertions fail loudly rather than silently corrupting every comparison.

---

## Metrics

`evaluate()` returns ROC-AUC, PR-AUC, Brier score, prevalence, distinct score count, and the headline metric.

### Headline: sensitivity at fixed testing capacity

*"If we can only test 10% of people, what fraction of true cases do we catch?"*

This is preferred over ROC-AUC because:

1. **It matches the actual decision.** The paper's stated purpose is prioritising scarce tests — a
   budget-constrained ranking problem, not a threshold-free one.
2. **AUC averages over operating points nobody uses.** It weights the 90%-capacity regime equally with the 5%
   regime. Nobody deploys at 90%.
3. **It is directly interpretable.** "Test 10% of arrivals, catch 73% of cases" is actionable. "AUC 0.8976" is not.
4. **AUC hides prevalence shift**, as the November results above demonstrate.
5. **It exposes ranking ties.** With ≤256 distinct scores, huge groups of people share one score. AUC smooths
   this away; `sensitivity_at_capacity` computes the expected value under random tie-breaking, so results
   don't silently depend on row ordering.

ROC-AUC is still reported for comparability with the paper — just don't lead with it.

---

## Replication notes and gotchas

Three issues were found while reproducing the published model. All are handled in code; they are documented
here because each one fails silently or confusingly.

**1. Windows line endings corrupt the released model.**
LightGBM splits trees using byte offsets in the `tree_sizes` header. A checkout with `core.autocrlf=true`
rewrites the file with CRLF, invalidating every offset — LightGBM then **aborts the interpreter** with
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

**Before the first deploy:** enable Pages under *Settings → Pages → Source: GitHub Actions*. Without it, the
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

