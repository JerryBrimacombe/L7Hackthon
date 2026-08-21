# Concepts and results

You do not need a machine-learning background to read the benchmark. The project asks a practical question:

> If only a fixed percentage of people can be tested, can we rank people so that more positive cases are tested first?

This is a **prioritisation** problem. The primary result is therefore not a yes/no diagnosis. It is the quality of the ordered list produced by a model.

## The basic workflow

1. The model learns from people tested during 22-31 March 2020.
2. It produces a risk score for each person in a later evaluation week.
3. People are sorted from highest score to lowest score.
4. The benchmark assumes only 10% can be tested and counts how many positive cases are included in that group.
5. The same scores are also assessed as probabilities and as a ranking across all possible cut-offs.

## Dates and leakage

The canonical benchmark is chronological:

| Name | Dates | Purpose |
| --- | --- | --- |
| `train_2020_03` | 22-31 March | Train the normal models |
| `test_2020_04` | 1-7 April | Main held-out evaluation week |
| `shift_2020_11` | 1-7 November | Temporal-shift evaluation |
| `train_2020_03_model` | 22-27 March | Model training for calibration experiments |
| `calibration_2020_03` | 28-31 March | Fit a probability calibrator only |

Never use an evaluation week to tune a model or its calibration. That would make the reported result optimistic.

## The two tracks

The source data contains the literal text `None` for some age and gender values. The project keeps two interpretations separate:

- **`paper`**, the default: drops rows with unknown age or gender and reproduces the published study.
- **`inclusive`**: keeps those rows and adds `Age_60_unknown` and `Gender_unknown` indicators.

The tracks contain different people and have different empirical ceilings. Compare models **within one track only**.

## Reading the leaderboard

- **Sensitivity at capacity**: with 10% testing capacity, the fraction of all true positive cases found in the highest-ranked 10% of people. Higher is better.
- **ROC-AUC**: how consistently positives are ranked above negatives over all possible cut-offs. `0.5` is random ordering and `1.0` is perfect ordering.
- **PR-AUC**: ranking quality expressed through precision and recall. It is especially useful when positives are uncommon.
- **Brier score**: average squared error of the predicted probabilities. Lower is better.
- **Log loss**: penalises confident wrong probabilities more heavily than Brier score. Lower is better.
- **Calibration error**: average gap between predicted probability and observed positive rate in equal-sized groups. Lower is better.
- **Calibration slope**: `1.0` is ideal. A value below `1.0` usually means predictions are too extreme; a value above `1.0` usually means they are too compressed.
- **Percentage of ceiling**: sensitivity compared with the `ceiling_lookup` reference for the same track. It is not a universal theoretical limit outside this dataset.

## Ranking versus probability

A ranking score answers “who should be considered first?” A calibrated probability answers “among people scored near 20%, are about 20% actually positive?” These are different requirements.

Sigmoid calibration is monotonic, so it normally changes probability interpretation without changing the order. Isotonic calibration is more flexible but can map many scores to the same value, which can change tie handling and ranking metrics.

## The ceiling lookup

There are only eight binary features in the paper track, so there are at most $2^8 = 256$ possible input patterns. `ceiling_lookup` assigns each pattern its observed positive rate in the training data. It is a useful empirical reference, not a guarantee of future performance: sparse patterns and temporal changes can make it optimistic.

## Result files

Each run writes a JSON file named like:

```text
results/logreg__paper__test_2020_04__20260821T120000+0000.json
```

Important fields include:

- `metrics`: the scores shown in the leaderboard.
- `features`: the input columns used.
- `track`: the population definition.
- `eval_split`: the evaluation week.
- `score_table`: counts of people and positives for each distinct predicted score.
- `calibration_method`: `sigmoid`, `isotonic`, or `null` for raw results.

The score table is compact and is enough to rebuild the ranking curves without storing every individual prediction.
