# AGENTS.md

## Purpose

This repository is a partial, reproduction-oriented implementation of the paper:

- [Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference](https://arxiv.org/abs/2509.04169)

Make sure to use that paper as a context when writing code for this repository. Link every parameter and methodology decisions back to that paper.

It is not the original paper codebase. The goal here is to maintain a runnable baseline for record-level LiRA on time-series forecasting, with enough structure to validate results, extend the implementation, and run experiments locally or on rented compute.

## Current scope

What this repo currently covers:

- LSTM forecasting target model
- record-level LiRA attack pipeline
- user-based dataset partitioning
- ELD preprocessing utilities
- synthetic smoke tests
- paper-style result summaries via `results.json` and `paper_report.md`

What it does not currently try to cover:

- every dataset from the paper
- every forecasting architecture from the paper
- every attack variant from the paper
- the original paper's exact code release or training environment

## Current implementation status

Main entrypoints:

- `src/pipeline/run_lira.py`: end-to-end experiment runner
- `src/pipeline/train_target.py`: target training and evaluation
- `src/attacks/lira.py`: LiRA signal and scoring utilities
- `src/data/eld.py`: ELD loading, filtering, and preprocessing
- `bootstrap_and_run.sh`: fresh-machine setup and run helper
- `flash_worker.py`: experimental Runpod Flash endpoint

Implemented behavior:

- user-group dataset splitting rather than mixed random record splits
- configurable LiRA runs with target and shadow training
- offline and online ELD/LSTM record-level presets
- synthetic end-to-end smoke validation
- ELD raw-format preprocessing and diagnostics

## Environment expectations

The active environment target is:

- Python `3.10` to `3.12`
- local pin: `3.12.10`
- recommended virtualenv: `.venv312`

The old Python `3.8` environment was kept only for earlier project history. New work should assume the Python 3.12 environment.

## ELD-specific note

This repo currently centers on the ELD record-level path.

The code supports a strict paper-style ELD preprocessing flow, but the public Kaggle mirror we used for testing does not preserve the paper's literal mean-usage filter outcome. Applying the strict absolute mean filter retains too few households for the intended `100`-household experiment.

Because of that, the current practical ELD baseline uses a documented fallback:

- keep the active-span trimming
- keep the hourly aggregation
- keep the minimum-length filtering
- keep the final `100`-household target
- replace the absolute mean threshold with selection of the central `100` households by mean usage after filtering

That fallback is implemented as the main practical preset for the public Kaggle mirror.

## Recommended execution flow

For a fresh machine:

1. Create the Python environment.
2. Install dependencies.
3. Download the Kaggle ELD mirror into a stable local path.
4. Check CUDA visibility.
5. Run a synthetic smoke test.
6. Launch the full ELD fallback run only after the smoke test passes.

The repo helper for that flow is:

- `bootstrap_and_run.sh`

Recommended commands:

```bash
bash bootstrap_and_run.sh
MODE=full bash bootstrap_and_run.sh
```

## Paper-facing claim boundary

The strongest safe characterization of this repository is:

- a credible partial reproduction
- a runnable implementation baseline
- a paper-aligned record-level LiRA pipeline with a documented ELD dataset-selection deviation when using the public Kaggle mirror

Avoid describing this repo as:

- the original paper implementation
- a full reproduction of every experiment in the paper
- an exact reproduction of the original ELD setup without qualification

If the fallback ELD run produces similar results to the paper, the best framing is:

- the repo reproduces the experimental structure and result trend credibly
- the implementation is suitable as a baseline for further verification and extension

## Flash status

Runpod Flash support was added experimentally in:

- `flash_worker.py`

This is not currently the recommended primary execution path for long experiments. Flash worked as a local development concept, but remote environment initialization was unreliable in practice. For serious runs, a full Pod or equivalent machine is the safer option.

## Practical defaults

These are the working defaults the repo is currently organized around:

- forecasting model: LSTM
- attack setting: record-level LiRA
- signal: MSE
- ELD fallback preset: `eld_lstm_record_lira_offline_mse_kaggle_fallback`
- smoke validation before full runs

## Maintenance guidance

When extending this repo:

- keep the README and this file aligned
- document any new dataset-specific deviations explicitly
- preserve smoke-testability
- prefer practical reproducibility over paper-theory exposition here
- cite the paper instead of duplicating its methodological explanation in repo docs