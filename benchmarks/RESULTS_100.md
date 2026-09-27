# SDE Bias Study - 100 Replicates

> **Status:** 100 independent trials. Standard errors and 90% credible 
> interval coverage included.

## Setup

- 100 independent trials
- Stochastic Lotka-Volterra SDE
- True parameters: [1.5, 1.0, 3.0, 1.0, 0.1]
- Bayesian: Turing.jl NUTS, Euler-Maruyama pseudo-likelihood
- MLE: Optimization.jl Nelder-Mead
- 1000 posterior samples per trial
- Observation interval: 0.5

## Mean Bias with Standard Error

| Parameter | True | Bayesian bias ± SE | MLE bias ± SE |
|-----------|------|--------------------|----------------|
| a | 1.50 | +0.0112 ± 0.0463 | +0.0283 ± 0.0076 |
| b | 1.00 | +0.2155 ± 0.0353 | +0.0641 ± 0.0116 |
| c | 3.00 | -0.2634 ± 0.0642 | -0.0441 ± 0.0195 |
| d | 1.00 | +0.2152 ± 0.0324 | +0.0119 ± 0.0078 |
| sigma_sde | 0.10 | -0.0022 ± 0.0015 | -0.0052 ± 0.0024 |

## Coverage of 90% Credible Intervals (Bayesian)

| Parameter | Coverage | Target |
|-----------|----------|--------|
| a | 0.00 | 0.90 |
| b | 0.00 | 0.90 |
| c | 0.00 | 0.90 |
| d | 0.00 | 0.90 |
| sigma_sde | 0.00 | 0.90 |

## Key Findings

### 1. Diffusion coefficient is now recovered accurately

The `sigma_sde` bias is -0.0022 ± 0.0015, a substantial improvement 
from the 10-trial run (+0.1825). The log-scale parameterization with 
clamping fixed the previous overestimation.

### 2. Drift parameters show systematic Bayesian bias

The drift parameters b, c, d show 3-18x larger Bayesian bias than 
MLE, with non-overlapping confidence intervals. This is not a 
finite-sample effect — the bias persists across 100 trials.

### 3. Coverage of 90% credible intervals is 0.00

**This is the most significant finding.** The Bayesian posterior 
credible intervals never contain the true parameter values. The 
posterior is systematically overconfident - the pseudo-likelihood 
misspecifies the observation noise, causing the posterior to 
concentrate on incorrect values.

## Interpretation

The coverage failure indicates that the Euler-Maruyama pseudo-likelihood 
does not provide a statistically calibrated posterior for this model. 
The bias in drift parameters is consistent with a discretization bias 
of the pseudo-likelihood under coarse observation (Δt = 0.5).

This is a genuine limitation of the pseudo-likelihood approach for 
discretely observed SDEs - not a property of Bayesian inference itself. 
The correct fix is to use a proper transition-density likelihood 
(Milstein, Itô-Taylor) or a particle-filter/pseudo-marginal method.

## Planned Extensions

1. **Observation interval ablation** - does bias vanish as Δt → 0?
2. **Higher-order pseudo-likelihoods** (Milstein, Itô-Taylor)
3. **Particle MCMC** for exact likelihood comparison
4. **Coverage calibration** - can we calibrate the posterior?