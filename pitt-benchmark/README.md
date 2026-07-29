# Pitt benchmark

This benchmark evaluates the existing frozen Dopa AdsTrace metric heads on the
Pitt Video Ads viewer-effectiveness dataset.

The Dopa models predict ROI, CVR, mean iCTR, and max iCTR. Pitt supplies mean
human effectiveness ratings from 1 to 5. Because those are different targets,
the zero-shot evaluation measures correlation, ranking agreement, and
same-topic winner accuracy rather than subtracting raw CTR from a Pitt rating.

The benchmark also compares three inexpensive calibration options:

1. Training-set mean
2. An affine Ridge calibration of predicted mean iCTR
3. A Ridge calibration of all four frozen Dopa outputs

The calibration choice is made on the Pitt validation split. The Dopa metric
heads and TRIBE feature extractors remain frozen.

## Data contract

The benchmark expects aligned Pitt feature directories containing:

```text
manifest.parquet
compact_features.parquet
```

The video and text+audio directories must contain identical ad IDs and splits.
Their 900-feature schemas must match the base feature schema in
`data/combined.parquet`.

## Run

From the repository root:

```powershell
python pitt-benchmark\benchmark_pitt_model.py `
  --video-dataset-dir C:\path\to\PittVideoAdsTribeV2_video `
  --text-audio-dataset-dir C:\path\to\PittVideoAdsTribeV2_text_audio `
  --output-dir pitt-benchmark\results
```

The model repository defaults to the current repository root. Use
`--metric-model-repo` only when running the script from a different checkout.

## Outputs

- `results/frozen_metric_predictions.parquet`: four frozen Dopa predictions for
  every Pitt ad.
- `results/calibrated_predictions.parquet`: actual rating, calibrated rating,
  prediction source, and signed error.
- `results/pitt_calibrator.joblib`: lightweight calibrator refit on
  train+validation.
- `results/metrics.json`: source-domain sanity metrics, zero-shot Pitt results,
  all calibration candidates, held-out metrics, confidence intervals, and
  limitations.

## Completed 160-ad result

The source-domain sanity check confirms that the saved mean-iCTR model still
has signal on the AdsTrace test split:

- Mean-iCTR Spearman: `0.459`
- Mean-iCTR R²: `0.187`

It does not transfer reliably to Pitt human-effectiveness ratings:

- All 160 Pitt ads, mean-iCTR Spearman: `0.019`
- Pitt 20-ad test split, mean-iCTR Spearman: `-0.338`
- Pitt test median-tier accuracy: `30%`

The validation procedure selected the training-mean baseline. None of the
frozen Dopa outputs improved validation RMSE. The baseline test MAE was `0.578`
rating points and test RMSE was `0.675`.

This is an exploratory pilot. A production effectiveness claim requires more
Pitt examples and a new untouched holdout.

## Tests

```powershell
python -m pytest pitt-benchmark\tests -q
```
