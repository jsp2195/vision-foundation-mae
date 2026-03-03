"""Greedy 1-step expected information gain stimulus selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_process import GaussianProcessPrior
from laplace_inference import LaplaceGPInference
from psychometric import LogisticPsychometric


@dataclass(frozen=True)
class Stimulus:
    freq_idx: int
    amplitude_db: float


@dataclass(frozen=True)
class SelectionConfig:
    amplitude_window_db: float
    amplitude_levels: int


class GreedyInformationGainSelector:
    def __init__(self, config: SelectionConfig, psychometric: LogisticPsychometric):
        self.config = config
        self.psychometric = psychometric

    def candidate_amplitudes(self, center: float) -> np.ndarray:
        offsets = np.linspace(-self.config.amplitude_window_db, self.config.amplitude_window_db, self.config.amplitude_levels)
        return center + offsets

    def select(self, inference: LaplaceGPInference) -> tuple[Stimulus, float]:
        current_entropy = GaussianProcessPrior.entropy(inference.cov)
        best_ig = -np.inf
        best_stimulus = Stimulus(freq_idx=0, amplitude_db=float(inference.mu[0]))

        for idx, mean_thr in enumerate(inference.mu):
            for amp in self.candidate_amplitudes(float(mean_thr)):
                p_detect = float(self.psychometric.detect_probability(float(amp), float(inference.mu[idx])))
                _, cov_yes = inference.hypothetical_update(freq_idx=idx, amplitude_db=float(amp), response=1)
                _, cov_no = inference.hypothetical_update(freq_idx=idx, amplitude_db=float(amp), response=0)
                ent_yes = GaussianProcessPrior.entropy(cov_yes)
                ent_no = GaussianProcessPrior.entropy(cov_no)
                expected_entropy = p_detect * ent_yes + (1.0 - p_detect) * ent_no
                ig = current_entropy - expected_entropy
                if ig > best_ig:
                    best_ig = ig
                    best_stimulus = Stimulus(freq_idx=idx, amplitude_db=float(amp))

        return best_stimulus, float(best_ig)
