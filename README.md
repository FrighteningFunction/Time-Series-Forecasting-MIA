# Time-Series-Forecasting-MIA

Subset implementation of https://arxiv.org/abs/2509.04169

## What this repo runs

This repository now contains a complete end-to-end pipeline for:

- training an LSTM forecasting target model,
- training LiRA shadow models on auxiliary data,
- scoring member vs non-member windows with prediction-error signals,
- reporting attack metrics such as ROC AUC and TPR at low FPR.

The implementation is still a subset of the paper:

- it uses an LSTM forecaster,
- it uses the MSE attack signal,
- it supports ELD-style 2D arrays (`time x user` by default, or `user x time` with `--series-axis 0`),
- it now splits datasets by user groups rather than mixing windows across users,
- it provides a synthetic sanity-check mode when you do not want to depend on dataset files yet.

## Quick start

Run a toy end-to-end check:

```bash
./.venv/bin/python -m src.pipeline.run_lira --synthetic --epochs 5 --num-shadow 3
```

Run on an ELD-style file:

```bash
./.venv/bin/python -m src.pipeline.run_lira \
  --data-path /path/to/eld.npy \
  --epochs 20 \
  --num-shadow 8 \
  --num-train-users 20 \
  --num-val-users 20 \
  --num-test-users 20 \
  --num-aux-users 40 \
  --output-dir artifacts/eld_lira
```

The command writes metrics to `results.json` under the chosen output directory.

## Input format

The data loader expects a 2D array:

- `.npy`: one array
- `.npz`: one stored array
- text or `.csv`: numeric matrix loaded with `numpy.loadtxt`

By default the code assumes `time x user`. If your file is `user x time`, pass:

```bash
--series-axis 0
```

## Paper-aligned notes

This code follows the subset described in `agents.md` and is intended for partial verification rather than full reproduction. In particular:

- windowing defaults are `L=100`, `H=20`,
- training uses MAE,
- LiRA uses Gaussian fits over per-window MSE signals,
- the current pipeline reports record-level attack metrics under user-based data partitioning,
- the shadow-model count is configurable, but defaults to `8` rather than the larger paper-scale setting.
