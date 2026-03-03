"""Laplace posterior inference for GP-threshold audiometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gaussian_process import GaussianProcessPrior
from psychometric import LogisticPsychometric

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrialObservation:
    freq_idx: int
    amplitude_db: float
    response: int


class LaplaceGPInference:
    def __init__(self, gp: GaussianProcessPrior, psychometric: LogisticPsychometric):
        self.gp = gp
        self.psychometric = psychometric
        self.mu = gp.mean.copy()
        self.cov = gp.covariance.copy()
        self.observations: list[TrialObservation] = []

    def add_observation(self, obs: TrialObservation) -> None:
        self.observations.append(obs)

    def _neg_log_posterior(self, theta: Array) -> float:
        delta = theta - self.gp.mean
        prior_term = 0.5 * float(delta.T @ self.gp.precision @ delta)
        ll = 0.0
        for obs in self.observations:
            ll += self.psychometric.log_likelihood(obs.response, obs.amplitude_db, float(theta[obs.freq_idx]))
        return prior_term - ll

    def _grad_and_hess(self, theta: Array) -> tuple[Array, Array]:
        delta = theta - self.gp.mean
        grad = self.gp.precision @ delta
        hess = self.gp.precision.copy()
        for obs in self.observations:
            idx = obs.freq_idx
            p = float(self.psychometric.detect_probability(obs.amplitude_db, float(theta[idx])))
            grad[idx] += self.psychometric.beta * (obs.response - p)
            hess[idx, idx] += (self.psychometric.beta**2) * p * (1.0 - p)
        return grad, hess

    def update(self, max_iter: int = 50, tol: float = 1e-6) -> None:
        if not self.observations:
            return
        theta = self.mu.copy()
        for _ in range(max_iter):
            grad, hess = self._grad_and_hess(theta)
            hess = 0.5 * (hess + hess.T) + np.eye(hess.shape[0], dtype=np.float64) * self.gp.config.jitter
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < tol:
                break
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Newton solve failed; Hessian singular.") from exc

            current = self._neg_log_posterior(theta)
            alpha = 1.0
            for _ in range(20):
                candidate = theta - alpha * step
                cand_val = self._neg_log_posterior(candidate)
                if cand_val <= current:
                    theta = candidate
                    break
                alpha *= 0.5
            else:
                break

        self.mu = theta
        _, hess = self._grad_and_hess(self.mu)
        hess = 0.5 * (hess + hess.T) + np.eye(hess.shape[0], dtype=np.float64) * self.gp.config.jitter
        try:
            chol = np.linalg.cholesky(hess)
            eye = np.eye(hess.shape[0], dtype=np.float64)
            y = np.linalg.solve(chol, eye)
            self.cov = np.linalg.solve(chol.T, y)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Hessian inversion failed; matrix likely singular.") from exc

    def hypothetical_update(self, freq_idx: int, amplitude_db: float, response: int) -> tuple[Array, Array]:
        shadow = LaplaceGPInference(self.gp, self.psychometric)
        shadow.mu = self.mu.copy()
        shadow.cov = self.cov.copy()
        shadow.observations = self.observations.copy()
        shadow.add_observation(TrialObservation(freq_idx=freq_idx, amplitude_db=amplitude_db, response=response))
        shadow.update()
        return shadow.mu, shadow.cov
