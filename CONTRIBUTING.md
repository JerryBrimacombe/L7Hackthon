# Contributing

Thank you for improving the benchmark. This project values reproducibility over leaderboard chasing: every change should make it easier to understand what was measured and why.

## Before you start

Read:

- [Getting started](docs/GETTING_STARTED.md) for setup and commands.
- [Concepts and results](docs/CONCEPTS.md) for the benchmark design.
- [Troubleshooting](docs/TROUBLESHOOTING.md) if a command does not work.

## Make a change

1. Create a branch for your change.
2. Make the smallest change that explains or fixes one thing.
3. Add or update a test when behavior changes.
4. Update the relevant documentation and command examples.
5. Run the full test suite before opening a pull request.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Add a model

Create a module under `covidbench/models/`. The registry discovers modules automatically. A model factory should return an object with `fit(X, y)` and `predict_proba(X)`:

```python
from .. import config
from ..registry import register


@register(
    "your_model",
    features=config.FEATURES_ALL,
    notes="Short explanation shown by --list",
    tracks=(config.COHORT_PAPER, config.COHORT_INCLUSIVE),
)
def your_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=config.RANDOM_SEED)
```

Use `tracks=(config.COHORT_PAPER,)` when a model cannot support the inclusive feature set. Optional dependencies are allowed; the runner skips unavailable models during `--all` runs.

Check that the model appears and runs:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --list
.\.venv\Scripts\python.exe -m covidbench.run --model your_model
```

## Update the generated website

The website is generated from stored results:

```powershell
.\.venv\Scripts\python.exe -m covidbench.compare --html --no-ci
```

Edit source text in `covidbench/compare.py` or chart code in `covidbench/plots.py`; do not hand-edit `docs/index.html`.

## Pull request checklist

- [ ] The change has a clear explanation for non-experts.
- [ ] Canonical temporal splits remain separate from research random splits.
- [ ] The `paper` and `inclusive` tracks are not compared as if they contain the same people.
- [ ] Tests pass locally.
- [ ] README or guide pages explain new commands, outputs, and caveats.
- [ ] Generated results or charts are regenerated when the change affects them.
