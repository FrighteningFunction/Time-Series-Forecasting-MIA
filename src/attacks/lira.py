import torch

import numpy as np


class GaussianEstimator:
    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean = float(np.mean(values))
        self.std = float(np.std(values) + 1e-8)

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / self.std
        return -0.5 * (z ** 2) - np.log(self.std) - 0.5 * np.log(2.0 * np.pi)


class LiRA:
    def __init__(self):
        self.in_dist = GaussianEstimator()
        self.out_dist = GaussianEstimator()

    def fit(self, member_signals, nonmember_signals):
        self.in_dist.fit(member_signals)
        self.out_dist.fit(nonmember_signals)

    def score(self, signals):
        signals = np.asarray(signals, dtype=np.float64)
        log_in = self.in_dist.logpdf(signals)
        log_out = self.out_dist.logpdf(signals)
        return log_in - log_out


def compute_mse_signal(model, loader, device):
    model.eval()
    signals = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            mse = torch.mean((y_hat - y) ** 2, dim=1)
            signals.extend(mse.cpu().numpy())

    return np.asarray(signals, dtype=np.float64)
