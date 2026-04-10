# AGENTS.md

## Overview

This codebase implements a **partial reproduction** of the paper. The goal is to reproduce some parts of the paper's results for the first time.

> Membership Inference Attacks on Time-Series Forecasting Models

Specifically, it reproduces the **LiRA (Likelihood Ratio Attack)** pipeline on the **ELD dataset** using an **LSTM forecasting model**.

The implementation focuses on correctness, modularity, and reusability for future integration (e.g., federated learning settings).

---

# What is Implemented

## 1. Time-Series Forecasting Pipeline (Paper Section: Model Training)

### Implemented in:
- `eld.py`
- `models/lstm.py`
- `pipeline/train_target.py`

### Description

- Sliding window generation:
  - Input: past `L` timesteps
  - Output: future `H` timesteps
- IQR normalization per time series
- LSTM-based forecasting model

This corresponds to the paper’s:
> Forecasting model training on time-series data

---

## 2. Signal Extraction (Paper Section: Attack Signals)

### Implemented in:
- `attacks/lira.py` (`compute_mse_signal`)

### Description

- Per-sample prediction error is used as the attack signal:

\[
s(x) = \frac{1}{H} \sum_{t=1}^{H} (y_t - \hat{y}_t)^2
\]

Where:
- \( y_t \) = true value at timestep \( t \)
- \( \hat{y}_t \) = predicted value at timestep \( t \)
- \( H \) = prediction horizon

This matches the paper’s use of:
> Prediction error (e.g., MSE) as a membership signal

---

## 3. Shadow Model Training (Paper Section: Shadow Models)

### Implemented in:
- `pipeline/run_lira.py`
  - `train_shadow_models`
  - `sample_shadow_split`

### Description

- Multiple shadow models are trained on **random subsets** of auxiliary data
- Each shadow model has:
  - its own training set (members)
  - its own validation set (non-members)

This corresponds to:
> Simulating the target model’s behavior using auxiliary data

---

## 4. Member vs Non-Member Distributions (Paper Core Idea)

### Implemented in:
- `run_lira`

### Description

For each shadow model:

- Training data → member signals
- Validation data → non-member signals

Aggregated across all shadow models to estimate:

\[
P(s \mid \text{member}) \quad \text{and} \quad P(s \mid \text{non-member})
\]

This is the core statistical foundation of the attack.

---

## 5. LiRA Attack (Paper Section: Likelihood Ratio Attack)

### Implemented in:
- `attacks/lira.py`

### Description

The attack computes:

\[
\text{score}(s) =
\log P(s \mid \text{member}) - \log P(s \mid \text{non-member})
\]

Where:
- \( s \) = signal (prediction error)
- Distributions are modeled as Gaussians:
  - \( \mu_{\text{in}}, \sigma_{\text{in}} \)
  - \( \mu_{\text{out}}, \sigma_{\text{out}} \)

Decision rule:
- Higher score → more likely to be a member

---

## 6. Target Model Attack (Paper Section: Attack Evaluation)

### Implemented in:
- `run_lira`

### Description

- Signals are computed for:
  - target training data (members)
  - unseen data (non-members)
- LiRA scores are computed for both groups

This corresponds to:
> Applying the trained attack to the target model

---

# What is NOT Yet Implemented (Deviations from Paper)

## 1. User-Level Splitting

### Current:
- Random sample-level split

### Paper:
- Split by **user / entity**

### Impact:
- Current implementation performs **record-level membership inference**
- Paper focuses on **user-level inference**

---

## 2. Number of Shadow Models

### Current:
- ~4–8 shadow models

### Paper:
- ~64 shadow models

### Impact:
- Lower statistical stability, but still valid

---

## 3. Multiple Signals

### Current:
- Only MSE signal

### Paper:
- Multiple signals (e.g., MAE, representation-based)

---

## 4. Multiple Architectures

### Current:
- LSTM only

### Paper:
- Multiple models (e.g., TS2Vec, others)

---

## 5. Federated Learning Setting

### Current:
- Centralized training

### Paper:
- Includes federated scenarios

---

# Summary of Coverage

| Component | Status |
|----------|--------|
| Forecasting model | ✅ Implemented |
| Signal extraction (MSE) | ✅ Implemented |
| Shadow models | ✅ Implemented |
| LiRA attack | ✅ Implemented |
| Target attack evaluation | ✅ Implemented |
| User-level split | ❌ Not implemented |
| Full-scale experiments | ❌ Not implemented |
| Federated setup | ❌ Not implemented |

---

# Interpretation

This codebase reproduces:

> The **core LiRA attack pipeline** for time-series forecasting models

while simplifying:
- data partitioning
- experimental scale

It is therefore a:

> **Valid partial reproduction of the paper’s main attack methodology**

---

# Future Extensions

- User-level splitting (paper-accurate threat model)
- Increased number of shadow models
- Additional signals
- Integration into federated learning frameworks

---
