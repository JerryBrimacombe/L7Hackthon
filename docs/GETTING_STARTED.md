# Getting started

This guide is for someone who has just downloaded or cloned the repository and wants to run the benchmark successfully.

## 1. Get the project

You can either clone the repository with Git or download it as a ZIP file from GitHub. If you download a ZIP file, extract it before opening the folder in VS Code.

The project folder is the folder containing `pyproject.toml`, `README.md`, and `covidbench/`. The commands below must be run from that folder.

## 2. Check Python

Use Python **3.13**. The pinned LightGBM dependency may not have a compatible wheel for newer Python versions.

```powershell
py -3.13 --version
```

On macOS or Linux, use:

```bash
python3.13 --version
```

If the command is not found, install Python 3.13 from [python.org](https://www.python.org/downloads/). On Windows, enable the option to add Python to PATH if the installer offers it.

## 3. Create an isolated environment

A virtual environment keeps this project’s packages separate from other Python projects.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-optional.txt
```

### macOS or Linux

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-optional.txt
```

The repository uses the Python executable inside `.venv` in all examples. This avoids problems caused by accidentally using a different Python installation.

## 4. Check the data

The April dataset (`v006`) is included in this repository, so the basic benchmark can run immediately.

The November temporal-shift benchmark needs the companion [nshomron/covidpred](https://github.com/nshomron/covidpred) repository. The easiest layout is:

```text
C:\Hackathon\
├── covidpred\       <- companion repository
└── L7Hackthon\      <- this repository
```

Clone it beside this project:

```powershell
git clone https://github.com/nshomron/covidpred.git ..\covidpred
```

On macOS or Linux, from the project folder:

```bash
git clone https://github.com/nshomron/covidpred.git ../covidpred
```

If the companion repository is somewhere else, set `COVIDPRED_ROOT` to its absolute path before running the commands. For example, in PowerShell:

```powershell
$env:COVIDPRED_ROOT = 'C:\data\covidpred'
```

## 5. Run the checks

Run the test suite first. A successful run ends with a summary such as `20 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```bash
.venv/bin/python -m pytest -q
```

## 6. Run one model

Start with the transparent logistic-regression baseline. This is quicker and easier to understand than running every model.

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --model logreg
```

```bash
.venv/bin/python -m covidbench.run --model logreg
```

The command prints sensitivity at the 10% testing capacity, ROC-AUC, PR-AUC, and the number of distinct scores. It also writes a timestamped JSON file under `results/`.

## 7. Run the full April benchmark

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all
```

```bash
.venv/bin/python -m covidbench.run --all
```

Optional model libraries are allowed to be absent. If one is missing, the command prints `SKIPPED` for that model and continues with the others.

## 8. Build the leaderboard and charts

```powershell
.\.venv\Scripts\python.exe -m covidbench.compare --html
```

```bash
.venv/bin/python -m covidbench.compare --html
```

Open `docs/index.html` in a browser. The page contains the leaderboard, explanations, and charts. Use `--no-ci` for a faster report without bootstrap intervals.

## 9. Run the November shift evaluation

This step requires the companion repository because it uses dataset `v0083`.

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all --eval-split shift_2020_11 --no-verify
```

The `--no-verify` flag is needed because the shift dataset intentionally does not have the April row-count assertions.

## 10. Try calibration

Calibration changes the meaning of a score from “relative ranking score” toward “estimated probability.” It must be fitted on dates separate from both model training and evaluation.

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all --calibrate sigmoid
.\.venv\Scripts\python.exe -m covidbench.compare --html --no-ci
```

Use sigmoid first. Isotonic is available for comparison but can create tied scores:

```powershell
.\.venv\Scripts\python.exe -m covidbench.run --all --calibrate isotonic
```

## What success looks like

After a normal run you should see:

- JSON result files in `results/`.
- `docs/index.html` after running `compare --html`.
- PNG charts in `docs/charts/`.
- A leaderboard where sensitivity is reported at the configured 10% testing capacity.

If a command fails, see [Troubleshooting](TROUBLESHOOTING.md) before changing the code.
