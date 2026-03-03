"""Metrics for convergence and reliability."""

from __future__ import annotations

import numpy as np


def mean_posterior_variance(cov: np.ndarray) -> float:
    return float(np.mean(np.diag(cov)))


def response_consistency(responses: list[int]) -> float:
    if len(responses) < 2:
        return 1.0
    diffs = np.abs(np.diff(np.asarray(responses, dtype=np.float64)))
    return float(1.0 - np.mean(diffs))


def entropy_slope(entropy_curve: list[float]) -> float:
    if len(entropy_curve) < 2:
        return 0.0
    x = np.arange(len(entropy_curve), dtype=np.float64)
    y = np.asarray(entropy_curve, dtype=np.float64)
    slope = np.polyfit(x, y, deg=1)[0]
    return float(slope)


def reliability_score(cov: np.ndarray, responses: list[int], entropy_curve: list[float]) -> float:
    var_term = np.exp(-mean_posterior_variance(cov) / 25.0)
    consistency_term = (response_consistency(responses) + 1.0) / 2.0
    slope = entropy_slope(entropy_curve)
    entropy_term = np.clip(-slope / 0.1, 0.0, 1.0)
    score = 0.5 * var_term + 0.3 * consistency_term + 0.2 * entropy_term
    return float(np.clip(score, 0.0, 1.0))
