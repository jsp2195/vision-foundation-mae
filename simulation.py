"""Simulation loop for Bayesian active audiometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration import DeterministicCalibration
from gaussian_process import GaussianProcessPrior
from laplace_inference import LaplaceGPInference, TrialObservation
from metrics import mean_posterior_variance, reliability_score
from psychometric import LogisticPsychometric
from stimulus_selection import GreedyInformationGainSelector


@dataclass(frozen=True)
class SimulationConfig:
    max_trials: int
    variance_stop: float
    seed: int
    response_noise_std: float


class SyntheticUser:
    def __init__(self, thresholds: np.ndarray, beta: float, rng: np.random.Generator, response_noise_std: float = 0.0):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.beta = float(beta)
        self.rng = rng
        self.response_noise_std = float(response_noise_std)

    def respond(self, freq_idx: int, amplitude_db: float) -> int:
        noisy_thr = self.thresholds[freq_idx] + self.rng.normal(0.0, self.response_noise_std)
        p = 1.0 / (1.0 + np.exp(-self.beta * (amplitude_db - noisy_thr)))
        return int(self.rng.random() < p)


class PatternLibrary:
    @staticmethod
    def generate(pattern: str, frequencies_hz: np.ndarray) -> np.ndarray:
        x = np.log2(frequencies_hz / 1000.0)
        if pattern == "normal":
            return np.full_like(frequencies_hz, 15.0, dtype=np.float64)
        if pattern == "mild_loss":
            return 25.0 + 8.0 * (x + 1.0)
        if pattern == "high_freq_rolloff":
            return 15.0 + 20.0 * np.clip(x, 0.0, None)
        if pattern == "irregular":
            return 20.0 + 8.0 * np.sin(2.7 * x) + 5.0 * np.cos(1.2 * x)
        raise ValueError(f"Unknown pattern: {pattern}")


def run_active_simulation(
    inference: LaplaceGPInference,
    selector: GreedyInformationGainSelector,
    user: SyntheticUser,
    calibration: DeterministicCalibration,
    config: SimulationConfig,
) -> dict:
    entropy_curve = [GaussianProcessPrior.entropy(inference.cov)]
    posterior_trace = [inference.mu.copy()]
    cov_trace = [inference.cov.copy()]
    responses: list[int] = []
    selected_stimuli: list[tuple[int, float]] = []

    for _ in range(config.max_trials):
        stimulus, _ = selector.select(inference)
        effective_amp = calibration.effective_amplitude(stimulus.freq_idx, stimulus.amplitude_db)
        response = user.respond(stimulus.freq_idx, effective_amp)

        obs = TrialObservation(freq_idx=stimulus.freq_idx, amplitude_db=effective_amp, response=response)
        inference.add_observation(obs)
        inference.update()

        responses.append(response)
        selected_stimuli.append((stimulus.freq_idx, effective_amp))
        posterior_trace.append(inference.mu.copy())
        entropy_curve.append(GaussianProcessPrior.entropy(inference.cov))
        cov_trace.append(inference.cov.copy())

        if mean_posterior_variance(inference.cov) < config.variance_stop:
            break

    reliability = reliability_score(inference.cov, responses, entropy_curve)
    return {
        "mu": inference.mu.copy(),
        "cov": inference.cov.copy(),
        "posterior_trace": posterior_trace,
        "entropy_curve": entropy_curve,
        "responses": responses,
        "stimuli": selected_stimuli,
        "reliability": reliability,
        "cov_trace": cov_trace,
        "trials": len(responses),
    }
