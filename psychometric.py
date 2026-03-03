"""Psychometric response models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def expit(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class LogisticPsychometric:
    def __init__(self, beta: float, prob_clip: float = 1e-6):
        self.beta = float(beta)
        self.prob_clip = float(prob_clip)

    def detect_probability(self, amplitude_db: Array | float, threshold_db: Array | float) -> Array:
        logits = self.beta * (np.asarray(amplitude_db, dtype=np.float64) - np.asarray(threshold_db, dtype=np.float64))
        probs = expit(logits)
        return np.clip(probs, self.prob_clip, 1.0 - self.prob_clip)

    def log_likelihood(self, response: int, amplitude_db: float, threshold_db: float) -> float:
        p = float(self.detect_probability(amplitude_db, threshold_db))
        return np.log(p) if response == 1 else np.log(1.0 - p)
