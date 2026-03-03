"""Gaussian process prior utilities for Bayesian active audiometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GPPriorConfig:
    sigma_f: float
    length_scale: float
    sigma_noise: float
    jitter: float


class GaussianProcessPrior:
    def __init__(self, frequencies_hz: Array, mean_db: Array, config: GPPriorConfig):
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
        self.log_freq = np.log(self.frequencies_hz)
        self.mean = np.asarray(mean_db, dtype=np.float64)
        self.config = config
        self.covariance = self._build_covariance()
        self._chol = np.linalg.cholesky(self.covariance)
        self.precision = self._chol_solve(self._chol, np.eye(self.mean.size, dtype=np.float64))

    def _build_covariance(self) -> Array:
        diffs = self.log_freq[:, None] - self.log_freq[None, :]
        sqdist = diffs * diffs
        base = (self.config.sigma_f**2) * np.exp(-0.5 * sqdist / (self.config.length_scale**2))
        noise = (self.config.sigma_noise**2) * np.eye(self.log_freq.size, dtype=np.float64)
        jitter = self.config.jitter * np.eye(self.log_freq.size, dtype=np.float64)
        return (base + noise + jitter).astype(np.float64)

    @staticmethod
    def _chol_solve(chol: Array, b: Array) -> Array:
        y = np.linalg.solve(chol, b)
        return np.linalg.solve(chol.T, y)

    def solve_covariance(self, b: Array) -> Array:
        return self._chol_solve(self._chol, b)

    @staticmethod
    def entropy(covariance: Array) -> float:
        n = covariance.shape[0]
        chol = np.linalg.cholesky(covariance)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        return 0.5 * (n * np.log(2.0 * np.pi * np.e) + log_det)
