# Bayesian Active Audiometry Framework (MVP v1)

A complete, numerically stable, CLI-driven implementation of **Bayesian Active Audiometry** with:

- Logistic psychometric detection model
- Single global Gaussian Process prior over threshold curve
- Laplace approximation posterior updates
- Greedy one-step expected information gain (IG) stimulus selection
- Deterministic calibration layer (relative scaling + compensation + ambient floor)
- Simulation, validation, metrics, and visualization tooling

## Mathematical model

### Frequency grid

Frequencies are fixed and log-spaced in the speech/audiology range (default 250–8000 Hz). All kernel computations use log-frequency.

### Psychometric model

For frequency index `i`:

\[
P(y=1 \mid a, \theta_i) = \sigma\left(\beta(a - \theta_i)\right)
\]

- `θ_i`: threshold at frequency `i`
- `β`: fixed slope (configured; no inference)
- `a`: effective stimulus amplitude after deterministic calibration

No lapse-rate modeling is included in MVP v1.

### GP prior over threshold curve

\[
\theta \sim \mathcal{N}(\mu_0, K)
\]

with RBF kernel on log-frequency:

\[
K_{ij} = \sigma_f^2 \exp\left(-\frac{(\log f_i - \log f_j)^2}{2\ell^2}\right) + \sigma_{noise}^2\delta_{ij}
\]

plus configurable numerical jitter.

### Laplace approximation

Given dataset `D` of binary responses, the posterior is approximated by:

\[
p(\theta\mid D) \approx \mathcal{N}(\mu, \Sigma)
\]

where:

1. MAP estimate `μ` maximizes log posterior (warm-started from previous posterior mean)
2. Hessian `H` of negative log posterior is evaluated at MAP
3. Covariance is `Σ = H^{-1}` using Cholesky factorization

Numerical stability features:

- Double precision arrays
- Probability clamping away from 0/1
- Cholesky solves instead of explicit matrix inverse where possible
- Hessian symmetrization + jitter
- Failure-safe singularity checks

### Greedy expected information gain selection

For each candidate `(f_i, a)`:

1. Compute current entropy:
   \[
   H(\Sigma) = \frac{1}{2}\log |2\pi e\Sigma|
   \]
2. Compute hypothetical posterior covariances under `y=1` and `y=0`
3. Compute expected posterior entropy:
   \[
   \mathbb{E}[H_{new}] = p(y=1)H(\Sigma_1) + (1-p(y=1))H(\Sigma_0)
   \]
4. Information gain:
   \[
   IG = H_{current} - \mathbb{E}[H_{new}]
   \]

The highest-IG stimulus is selected. This is strictly **1-step greedy** (no lookahead).

## Phase-1 calibration layer

Calibration is deterministic and externally parameterized:

- Relative amplitude scaling
- Frequency compensation curve (`compensation_curve.json`)
- Ambient noise floor clipping

The module boundary is intentionally isolated so phase 2 can introduce latent calibration parameters (e.g., device gain, noise floor) into a joint posterior without refactoring core GP/Laplace code.

## Repository structure

- `gaussian_process.py`
- `laplace_inference.py`
- `psychometric.py`
- `stimulus_selection.py`
- `calibration.py`
- `simulation.py`
- `validation.py`
- `metrics.py`
- `visualization.py`
- `cli.py`
- `config.yaml`
- `requirements.txt`

## CLI usage

```bash
python cli.py run_simulation --config config.yaml --seed 42
python cli.py validate --config config.yaml --seed 42
python cli.py plot_results --config config.yaml
```

Outputs:

- Threshold estimates (`mu`)
- 95% confidence intervals (from posterior diagonal)
- Full posterior covariance matrix
- Reliability score `[0,1]`
- Convergence and entropy curves
- Plots:
  - Audiogram + uncertainty bands
  - Entropy reduction curve
  - MAE curve
  - Posterior variance heatmap

## Validation protocol

`validate` compares active Bayesian selection against a fixed staircase baseline with:

- MAE to ground truth
- Trials to convergence
- Entropy behavior (active run)

## Convergence interpretation

Session terminates if either:

- Mean posterior variance is below configured threshold, or
- Maximum trial budget is reached

Low posterior variance + sustained entropy reduction generally indicates reliable threshold recovery.

## Phase-2 roadmap

MVP v1 is explicitly prepared for **calibration-invariant joint inference** by:

1. Keeping a global latent vector interface in inference
2. Isolating calibration as a replaceable module
3. Maintaining covariance-aware entropy/selection logic

Phase 2 will append latent device parameters (gain/noise floor) and infer them jointly with thresholds in a hierarchical Bayesian model.

## Constraints honored

- No RL
- No GNN
- No lapse-rate inference
- No multi-step lookahead
- No joint device gain inference in v1
- CPU compatible
- Seed-reproducible
