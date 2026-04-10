import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


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
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1 + 1e-8
    return (x - q1) / iqr


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
