# Time-Series-Forecasting-MIA

Partial implementation of the experiments described in the paper:

- [Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference](https://arxiv.org/abs/2509.04169)

This repository is not the original code release for the paper. It is a reproduction-oriented implementation focused on getting a working record-level LiRA pipeline in place, validating it end to end, and making it runnable locally or on Runpod Flash.

## Status

What currently works:

- LSTM target-model training for time-series forecasting
- record-level LiRA attack pipeline
- user-based dataset splitting
- synthetic smoke tests
- ELD preprocessing from a raw matrix or raw text/CSV-style source
- paper-style reporting output via `results.json` and `paper_report.md`
- optional Runpod Flash deployment through `flash_worker.py`

What is intentionally incomplete or approximate:

- this repo does not aim to cover every dataset/model/attack combination from the paper
- the downloadable Kaggle ELD mirror does not satisfy the paper's literal mean-usage filter, so this repo includes a documented fallback household-selection rule for ELD
- remote artifact persistence is not implemented yet; API responses are more reliable than worker-local files for remote runs

## Environment

The project currently targets Python `3.10` to `3.12` and is pinned locally to `3.12.10`.

Recommended setup:

```bash
pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv312
./.venv312/bin/python -m pip install --upgrade pip
./.venv312/bin/python -m pip install -r requirements.txt
```

The previous `.venv` based on Python `3.8` was kept around only because it was used earlier in the project history. New work should use `.venv312`.

## Repository layout

- `src/pipeline/run_lira.py`: main experiment entrypoint
- `src/pipeline/train_target.py`: target-model training and evaluation
- `src/attacks/lira.py`: LiRA scoring utilities
- `src/data/eld.py`: ELD loading and preprocessing
- `flash_worker.py`: Runpod Flash endpoint for remote execution
- `agents.md`: repo-specific implementation notes and goals

## Quick start

Run a small synthetic smoke test:

```bash
./.venv312/bin/python -m src.pipeline.run_lira \
  --synthetic \
  --synthetic-users 10 \
  --synthetic-steps 240 \
  --epochs 2 \
  --num-shadow 2 \
  --num-runs 1 \
  --batch-size 64 \
  --num-train-users 2 \
  --num-val-users 2 \
  --num-test-users 2 \
  --num-aux-users 4 \
  --output-dir artifacts/paper_smoke_py312
```

If CUDA is available:

```bash
./.venv312/bin/python -m src.pipeline.run_lira \
  --synthetic \
  --synthetic-users 10 \
  --synthetic-steps 240 \
  --epochs 2 \
  --num-shadow 2 \
  --num-runs 1 \
  --batch-size 64 \
  --num-train-users 2 \
  --num-val-users 2 \
  --num-test-users 2 \
  --num-aux-users 4 \
  --device cuda \
  --output-dir artifacts/paper_smoke_py312_cuda
```

Outputs are written under the selected `output_dir`, typically including:

- `results.json`
- `paper_report.md`

## Fresh machine workflow

If you are starting from a new GPU machine and want the safest path with the least wasted paid time, use `bootstrap_and_run.sh`.

What it does:

- creates the Python environment if needed
- installs dependencies
- downloads the public Kaggle ELD mirror into `data/df_kwh_adjusted.csv`
- checks whether CUDA is visible to PyTorch
- runs a synthetic smoke test
- optionally launches the full ELD fallback experiment

Default bootstrap run:

```bash
bash bootstrap_and_run.sh
```

That performs setup, downloads the dataset if needed, checks GPU visibility, and runs only the smoke test.

To launch the full ELD fallback run after bootstrap:

```bash
MODE=full bash bootstrap_and_run.sh
```

Useful overrides:

```bash
PYTHON_BIN=python3.12 MODE=full DEVICE=cuda bash bootstrap_and_run.sh
DATA_PATH=/absolute/path/to/df_kwh_adjusted.csv MODE=full bash bootstrap_and_run.sh
SKIP_DOWNLOAD=1 SKIP_SMOKE=1 MODE=full bash bootstrap_and_run.sh
```

This script is the recommended machine-start sequence for rented compute because it verifies the environment before the expensive full run starts.

## ELD workflow

This repo currently centers on the ELD record-level pipeline.

Supported input shapes:

- `.npy`
- `.npz`
- text/CSV numeric matrices
- raw ELD-style delimited files via `--eld-raw-format`

By default, the loader expects `time x user`. If your file is `user x time`, pass:

```bash
--series-axis 0
```

### Strict paper-style ELD preprocessing

The code supports the paper-style preprocessing path:

- active-span trimming
- hourly aggregation
- minimum valid length filter
- mean-usage filtering
- truncation to a common length

However, the downloadable Kaggle mirror of ELD that we used for testing does not preserve the paper's literal mean-usage filter outcome. Applying the strict absolute mean filter keeps too few households for the intended 100-household experiment.

### Kaggle-compatible fallback

To keep the run usable without changing the rest of the pipeline, this repo includes a fallback ELD preset that:

- keeps the same length filtering
- keeps the same final `100`-household target
- replaces the absolute mean threshold with selection of the central `100` households by mean usage after filtering

This is the recommended practical run for the public Kaggle ELD mirror:

```bash
./.venv312/bin/python -m src.pipeline.run_lira \
  --paper-preset eld_lstm_record_lira_offline_mse_kaggle_fallback \
  --data-path /path/to/df_kwh_adjusted.csv \
  --output-dir artifacts/eld_kaggle_fallback
```

Interpretation:

- this is a reproduction-oriented run
- it is paper-aligned except for the documented ELD household-selection fallback

## Paper presets

The main presets currently wired into the runner are:

- `eld_lstm_record_lira_online_mse`
- `eld_lstm_record_lira_offline_mse`
- `eld_lstm_record_lira_offline_mse_kaggle_fallback`

These presets are implemented in `src/pipeline/run_lira.py` and are intended to reduce command-line drift when comparing runs.

## Local execution

General pattern:

```bash
./.venv312/bin/python -m src.pipeline.run_lira [options]
```

Useful knobs:

- `--device cpu|cuda|auto`
- `--epochs`
- `--num-shadow`
- `--num-runs`
- `--batch-size`
- `--output-dir`
- `--paper-preset`

If you want to inspect available arguments directly:

```bash
./.venv312/bin/python -m src.pipeline.run_lira --help
```

## Runpod Flash

This repo includes a Flash endpoint in `flash_worker.py`.

The current worker:

- targets `NVIDIA_GEFORCE_RTX_4090`
- defaults to the ELD Kaggle fallback preset
- can auto-download the Kaggle dataset on the worker if needed

Local Flash development:

```bash
source .venv312/bin/activate
flash login
flash run
```

Remote deployment:

```bash
source .venv312/bin/activate
flash app create tsf-mia
flash env create prod -a tsf-mia
flash deploy -a tsf-mia -e prod
```

Important note:

- `.flash/` is generated by Flash and should be treated as build output, not source code

## Current caveats

- the public Kaggle ELD mirror is usable, but not identical to the data behavior implied by the paper's literal ELD household filter
- remote worker-local files are not automatically uploaded after a Flash run
- the current focus is record-level LiRA; broader paper coverage remains future work

## Citation

If you use this repository, please cite the paper rather than this codebase as the original source of the method:

```bibtex
@article{zimmerman2025privacy,
  title={Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference},
  author={Zimmerman, William and others},
  journal={arXiv preprint arXiv:2509.04169},
  year={2025}
}
```
