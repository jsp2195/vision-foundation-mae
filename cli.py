"""CLI entrypoint for Bayesian Active Audiometry MVP v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from calibration import CalibrationConfig, DeterministicCalibration
from gaussian_process import GPPriorConfig, GaussianProcessPrior
from laplace_inference import LaplaceGPInference
from psychometric import LogisticPsychometric
from simulation import PatternLibrary, SimulationConfig, SyntheticUser, run_active_simulation
from stimulus_selection import GreedyInformationGainSelector, SelectionConfig
from validation import mae, run_fixed_staircase
from visualization import plot_audiogram, plot_entropy_curve, plot_mae_curve, plot_variance_heatmap


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_components(cfg: dict):
    frequencies = np.asarray(cfg["frequencies_hz"], dtype=np.float64)
    prior_mean = np.full(frequencies.size, float(cfg["prior_mean_db"]), dtype=np.float64)

    gp = GaussianProcessPrior(
        frequencies_hz=frequencies,
        mean_db=prior_mean,
        config=GPPriorConfig(**cfg["gp"]),
    )
    psychometric = LogisticPsychometric(**cfg["psychometric"])
    inference = LaplaceGPInference(gp=gp, psychometric=psychometric)
    selector = GreedyInformationGainSelector(config=SelectionConfig(**cfg["selection"]), psychometric=psychometric)

    cal_cfg = CalibrationConfig(
        relative_scale=cfg["calibration"]["relative_scale"],
        ambient_noise_floor_db=cfg["calibration"]["ambient_noise_floor_db"],
    )
    calibration = DeterministicCalibration.from_json(
        frequencies_hz=frequencies,
        config=cal_cfg,
        path=cfg["calibration"]["compensation_json"],
    )
    return frequencies, psychometric, inference, selector, calibration




def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def cmd_run_simulation(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    rng = np.random.default_rng(cfg["seed"])

    frequencies, psychometric, inference, selector, calibration = make_components(cfg)
    truth = PatternLibrary.generate(cfg["simulation"].get("pattern", "normal"), frequencies)
    user = SyntheticUser(truth, psychometric.beta, rng, cfg["simulation"]["response_noise_std"])

    sim_cfg = SimulationConfig(
        max_trials=cfg["simulation"]["max_trials"],
        variance_stop=cfg["simulation"]["variance_stop"],
        seed=cfg["seed"],
        response_noise_std=cfg["simulation"]["response_noise_std"],
    )
    results = run_active_simulation(inference, selector, user, calibration, sim_cfg)
    results["frequencies_hz"] = frequencies.tolist()
    results["truth"] = truth.tolist()
    results["mae"] = mae(results["mu"], truth)

    serializable = to_serializable(results)
    out_json = Path(cfg["outputs"]["results_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"Saved simulation results to {out_json}")
    print(f"Trials: {results['trials']}, MAE: {results['mae']:.3f} dB, Reliability: {results['reliability']:.3f}")


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    rng = np.random.default_rng(cfg["seed"])

    frequencies, psychometric, active_inference, selector, calibration = make_components(cfg)
    truth = PatternLibrary.generate(cfg["simulation"].get("pattern", "normal"), frequencies)

    user_active = SyntheticUser(truth, psychometric.beta, rng, cfg["simulation"]["response_noise_std"])
    sim_cfg = SimulationConfig(
        max_trials=cfg["simulation"]["max_trials"],
        variance_stop=cfg["simulation"]["variance_stop"],
        seed=cfg["seed"],
        response_noise_std=cfg["simulation"]["response_noise_std"],
    )
    active_results = run_active_simulation(active_inference, selector, user_active, calibration, sim_cfg)

    _, _, stair_inference, _, _ = make_components(cfg)
    user_stair = SyntheticUser(truth, psychometric.beta, np.random.default_rng(cfg["seed"] + 1), cfg["simulation"]["response_noise_std"])
    stair_results = run_fixed_staircase(
        inference=stair_inference,
        user=user_stair,
        frequencies_hz=frequencies,
        start_db=cfg["validation"]["staircase_start_db"],
        step_db=cfg["validation"]["staircase_step_db"],
        max_trials=cfg["simulation"]["max_trials"],
        variance_stop=cfg["simulation"]["variance_stop"],
    )

    active_mae = mae(active_results["mu"], truth)
    stair_mae = mae(stair_results["mu"], truth)
    print("Validation report")
    print("================")
    print(f"Active MAE: {active_mae:.3f} dB")
    print(f"Staircase MAE: {stair_mae:.3f} dB")
    print(f"Active trials: {active_results['trials']}")
    print(f"Staircase trials: {stair_results['trials']}")


def cmd_plot_results(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    results_path = Path(cfg["outputs"]["results_json"])
    if not results_path.exists():
        raise FileNotFoundError("Run simulation first to generate results JSON.")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    frequencies = np.asarray(data["frequencies_hz"], dtype=np.float64)
    mu = np.asarray(data["mu"], dtype=np.float64)
    cov = np.asarray(data["cov"], dtype=np.float64)
    truth = np.asarray(data.get("truth"), dtype=np.float64) if "truth" in data else None
    entropy_curve = [float(x) for x in data["entropy_curve"]]

    posterior_trace = [np.asarray(x, dtype=np.float64) for x in data.get("posterior_trace", [])]
    mae_curve = [mae(x, truth) for x in posterior_trace] if truth is not None and posterior_trace else []

    cov_trace = [np.asarray(c, dtype=np.float64) for c in data.get("cov_trace", [cov])]

    plot_dir = Path(cfg["outputs"]["plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_audiogram(frequencies, mu, cov, truth, str(plot_dir / "audiogram.png"))
    plot_entropy_curve(entropy_curve, str(plot_dir / "entropy_curve.png"))
    if mae_curve:
        plot_mae_curve(mae_curve, str(plot_dir / "mae_curve.png"))
    plot_variance_heatmap(cov_trace, str(plot_dir / "variance_heatmap.png"))
    print(f"Plots saved to {plot_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian Active Audiometry MVP v1")
    parser.add_argument("command", choices=["run_simulation", "validate", "plot_results"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run_simulation":
        cmd_run_simulation(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "plot_results":
        cmd_plot_results(args)


if __name__ == "__main__":
    main()
