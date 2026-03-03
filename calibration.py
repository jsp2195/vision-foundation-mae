"""Deterministic calibration layer for phase-1 active audiometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationConfig:
    relative_scale: float
    ambient_noise_floor_db: float


class DeterministicCalibration:
    """Deterministic mapping from requested stimulus amplitude to effective amplitude.

    Interface intentionally isolates calibration transform so future versions can
    replace deterministic parameters with latent inferred variables.
    """

    def __init__(self, frequencies_hz: Array, config: CalibrationConfig, compensation_curve_db: dict[str, float]):
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
        self.config = config
        self.compensation = np.array(
            [float(compensation_curve_db.get(str(int(f)), compensation_curve_db.get(str(float(f)), 0.0))) for f in self.frequencies_hz],
            dtype=np.float64,
        )

    @classmethod
    def from_json(cls, frequencies_hz: Array, config: CalibrationConfig, path: str | Path) -> "DeterministicCalibration":
        with open(path, "r", encoding="utf-8") as f:
            curve = json.load(f)
        return cls(frequencies_hz=frequencies_hz, config=config, compensation_curve_db=curve)

    def effective_amplitude(self, freq_idx: int, requested_db: float) -> float:
        scaled = self.config.relative_scale * float(requested_db)
        compensated = scaled + float(self.compensation[freq_idx])
        return max(compensated, self.config.ambient_noise_floor_db)
