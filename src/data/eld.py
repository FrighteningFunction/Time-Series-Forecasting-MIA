import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_windows(series, L=100, H=20):
    X, Y = [], []
    T = len(series)

    for t in range(T - L - H + 1):
        x = series[t:t+L]
        y = series[t+L:t+L+H]

        X.append(x)
        Y.append(y)

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def iqr_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    # RobustScaler-style centering/scaling is much safer here than dividing by
    # a near-zero IQR. Sparse ELD households often have q1 == q3 == 0
    if iqr < 1e-6:
        std = float(np.std(x))
        scale = std if std >= 1e-6 else 1.0
    else:
        scale = float(iqr)

    return (x - median) / scale


def preprocess_eld_matrix(
    matrix,
    aggregate_factor=4,
    min_valid_steps=15000,
    min_mean=200.0,
    max_mean=2000.0,
    truncate_length=15000,
    mean_filter_mode="absolute",
    target_users=None,
):
    diagnostics = summarize_eld_preprocessing(
        matrix,
        aggregate_factor=aggregate_factor,
        min_valid_steps=min_valid_steps,
        min_mean=min_mean,
        max_mean=max_mean,
        truncate_length=truncate_length,
        mean_filter_mode=mean_filter_mode,
        target_users=target_users,
    )

    if not diagnostics["kept_series"]:
        raise ValueError("No ELD household survived the paper preprocessing filters.")

    min_length = min(len(series) for series in diagnostics["kept_series"])
    processed = np.stack(
        [series[:min_length] for series in diagnostics["kept_series"]],
        axis=1,
    )
    return processed.astype(np.float32)


def summarize_eld_preprocessing(
    matrix,
    aggregate_factor=4,
    min_valid_steps=15000,
    min_mean=200.0,
    max_mean=2000.0,
    truncate_length=15000,
    mean_filter_mode="absolute",
    target_users=None,
):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}.")

    original_shape = tuple(int(x) for x in matrix.shape)
    if aggregate_factor > 1:
        trimmed = (matrix.shape[0] // aggregate_factor) * aggregate_factor
        matrix = matrix[:trimmed]
        matrix = matrix.reshape(
            trimmed // aggregate_factor, aggregate_factor, matrix.shape[1]
        )
        matrix = matrix.sum(axis=1)

    candidates = []
    mean_values = []
    surviving_length_count = 0
    nonzero_count = 0

    for col_idx in range(matrix.shape[1]):
        series = matrix[:, col_idx]
        nonzero = np.flatnonzero(series > 0)
        if len(nonzero) == 0:
            continue
        nonzero_count += 1

        series = series[nonzero[0]:nonzero[-1] + 1]
        if len(series) < min_valid_steps:
            continue
        surviving_length_count += 1

        mean_usage = float(np.mean(series))
        mean_values.append(mean_usage)

        candidates.append(
            {
                "series": series[:truncate_length],
                "mean_usage": mean_usage,
                "original_index": int(col_idx),
            }
        )

    if mean_filter_mode == "absolute":
        kept_candidates = [
            item
            for item in candidates
            if min_mean <= item["mean_usage"] <= max_mean
        ]
    elif mean_filter_mode == "middle_n":
        if target_users is None:
            raise ValueError("target_users is required when mean_filter_mode='middle_n'.")
        if len(candidates) < target_users:
            raise ValueError(
                f"Need at least {target_users} households after length filtering, "
                f"but found {len(candidates)}."
            )
        ranked = sorted(candidates, key=lambda item: item["mean_usage"])
        start = (len(ranked) - target_users) // 2
        kept_candidates = ranked[start:start + target_users]
    elif mean_filter_mode == "none":
        kept_candidates = candidates
    else:
        raise ValueError(
            "mean_filter_mode must be one of: absolute, middle_n, none."
        )

    return {
        "original_shape": original_shape,
        "hourly_shape": tuple(int(x) for x in matrix.shape),
        "num_households_total": int(original_shape[1]),
        "num_nonzero_households": int(nonzero_count),
        "num_min_length_households": int(surviving_length_count),
        "num_mean_filtered_households": int(len(kept_candidates)),
        "min_valid_steps": int(min_valid_steps),
        "min_mean": float(min_mean),
        "max_mean": float(max_mean),
        "truncate_length": int(truncate_length),
        "mean_filter_mode": mean_filter_mode,
        "target_users": target_users,
        "mean_usage_stats": None if not mean_values else {
            "min": float(np.min(mean_values)),
            "p25": float(np.quantile(mean_values, 0.25)),
            "median": float(np.median(mean_values)),
            "p75": float(np.quantile(mean_values, 0.75)),
            "max": float(np.max(mean_values)),
        },
        "kept_series": [item["series"] for item in kept_candidates],
        "kept_original_indices": [item["original_index"] for item in kept_candidates],
        "kept_mean_usage_stats": None if not kept_candidates else {
            "min": float(np.min([item["mean_usage"] for item in kept_candidates])),
            "median": float(np.median([item["mean_usage"] for item in kept_candidates])),
            "max": float(np.max([item["mean_usage"] for item in kept_candidates])),
        },
    }


def load_eld_matrix(path, series_axis="auto", delimiter=","):
    if path.endswith(".npy"):
        matrix = np.load(path)
    elif path.endswith(".npz"):
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(
                "NPZ input must contain a single array or be converted to .npy/.csv first."
            )
        matrix = archive[archive.files[0]]
    else:
        matrix = np.loadtxt(path, delimiter=delimiter)

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}.")

    if series_axis == "auto":
        # ELD-style inputs are usually time x user. If the matrix is transposed,
        # the shorter dimension is often the number of users.
        series_axis = 1 if matrix.shape[0] >= matrix.shape[1] else 0

    if series_axis == 0:
        matrix = matrix.T
    elif series_axis != 1:
        raise ValueError("series_axis must be one of: auto, 0, 1.")

    return matrix


def load_eld_raw_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            if not _is_float(parts[1].replace(",", ".")):
                continue
            rows.append([float(value.replace(",", ".")) for value in parts[1:]])

    if not rows:
        raise ValueError(f"No numeric household data could be parsed from {path}.")

    return np.asarray(rows, dtype=np.float32)


def build_user_datasets(matrix, L=100, H=20):
    user_datasets = []

    for user_idx in range(matrix.shape[1]):
        series = iqr_normalize(matrix[:, user_idx])
        X, Y = create_windows(series, L=L, H=H)
        if len(X) == 0:
            continue
        user_datasets.append(TimeSeriesDataset(X, Y))

    if not user_datasets:
        raise ValueError(
            f"No windows could be created with L={L}, H={H} for matrix shape {matrix.shape}."
        )

    return user_datasets


def concat_datasets(datasets):
    if not datasets:
        raise ValueError("Expected at least one dataset to concatenate.")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
