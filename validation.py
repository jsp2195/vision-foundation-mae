"""Validation routines comparing active audiometry against fixed staircase baseline."""

from __future__ import annotations

import numpy as np

from laplace_inference import LaplaceGPInference, TrialObservation
from metrics import mean_posterior_variance
from simulation import SyntheticUser


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run_fixed_staircase(
    inference: LaplaceGPInference,
    user: SyntheticUser,
    frequencies_hz: np.ndarray,
    start_db: float,
    step_db: float,
    max_trials: int,
    variance_stop: float,
) -> dict:
    amps = np.full_like(frequencies_hz, start_db, dtype=np.float64)
    direction = np.full_like(frequencies_hz, -1.0, dtype=np.float64)
    responses: list[int] = []
    trial = 0
    while trial < max_trials and mean_posterior_variance(inference.cov) > variance_stop:
        for idx in range(frequencies_hz.size):
            response = user.respond(idx, float(amps[idx]))
            obs = TrialObservation(freq_idx=idx, amplitude_db=float(amps[idx]), response=response)
            inference.add_observation(obs)
            inference.update()
            responses.append(response)
            direction[idx] = -1.0 if response == 1 else 1.0
            amps[idx] += direction[idx] * step_db
            trial += 1
            if trial >= max_trials or mean_posterior_variance(inference.cov) <= variance_stop:
                break

    return {
        "mu": inference.mu.copy(),
        "cov": inference.cov.copy(),
        "responses": responses,
        "trials": trial,
    }
