# Troubleshooting

## `python` or `py -3.13` is not found

Install Python 3.13 and create the environment again. Check that the command points to the expected version:

```powershell
py -3.13 --version
```

Use the Python executable inside `.venv` for project commands rather than a global `python` command.

## `No module named covidbench`

The command is probably running outside the repository root or outside the virtual environment. Change directory to the folder containing `pyproject.toml` and use:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --list
```

## `FileNotFoundError` for dataset `v0083`

The November dataset is not included in this repository. Clone [nshomron/covidpred](https://github.com/nshomron/covidpred) beside `L7Hackthon`, or set `COVIDPRED_ROOT` to the folder containing its `data/` directory.

The April benchmark can still run using the v006 CSV included at the repository root.

## `Could not find corona_tested_individuals...`

Check all of the following:

1. You are running the command from the project root.
2. The file is either in the repository root, under `data/`, or in the companion `covidpred/data/` folder.
3. `COVIDPRED_ROOT` points to the companion repository itself, not its `data` subfolder.

## A model prints `SKIPPED (ModuleNotFoundError...)`

That model uses an optional package. Install the optional requirements if you want every model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

The benchmark is designed to continue with the models that are available.

## A cohort verification assertion fails

The loader checks known row counts and positive counts to detect preprocessing drift. This often means that a data file, date range, or missing-value rule changed. Do not use `--no-verify` to hide the problem on the canonical April benchmark. Use it only for the November shift split, whose counts are intentionally not fixed in the same way.

## The leaderboard says there are no results

Run at least one model first:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --model logreg
.\.venv\Scripts\python.exe -m covidbench.compare --html
```

The comparison command reads JSON files from `results/`; it does not train models itself.

## Charts are missing

Charts are generated only when result files contain `score_table` data. Rerun the benchmark and then:

```powershell
.\.venv\Scripts\python.exe -m covidbench.compare --html
```

Open `docs/index.html` after the command completes.

## The calibration command changes the model name

This is expected. Calibrated results are saved with names such as `logreg__calibrated_sigmoid` so raw and calibrated results can be compared rather than silently replacing one another.

## The calibration curve is not perfect

Calibration is fitted on a separate March period and evaluated on April or November. A later change in prevalence can make an old probability mapping inaccurate. Calibration improves probability interpretation; it is not a promise that future populations will have the same base rate.

## Windows LightGBM or line-ending problems

The released model artifacts are text files. If you modify or replace them, keep their expected feature names and avoid changing their line endings. The loader normalises CRLF line endings when reading the released artifacts.

## Tests fail after editing model or data code

Run the focused checks first, then the complete suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_pipeline.py
.\.venv\Scripts\python.exe -m pytest -q
```

Read the first failure rather than the last one. The cohort tests are intended to catch changes that would invalidate historical comparisons.
