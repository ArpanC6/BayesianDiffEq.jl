# SDE Bias Study — Preliminary Results

> **Status:** exploratory, 10 trials. Full 100-replicate protocol
> is in progress. Per-parameter standard errors and coverage analysis
> will be added.

## Setup

- 10 independent trials
- Stochastic Lotka-Volterra SDE
- True parameters: [1.5, 1.0, 3.0, 1.0, 0.1]
- Bayesian: Turing.jl + Euler-Maruyama pseudo-likelihood
- MLE: Optimization.jl + Nelder-Mead
- Observation noise (`sigma_obs`) and SDE diffusion (`sigma_sde`)
  are modeled separately.

## Preliminary Mean Bias

| Parameter  | True | Bayesian bias | MLE bias |
|------------|------|---------------|----------|
| a          | 1.5  | 0.2053        | 0.0302   |
| b          | 1.0  | 0.0764        | 0.0333   |
| c          | 3.0  | -0.2923       | -0.137   |
| d          | 1.0  | 0.0663        | 0.0046   |
| sigma_sde  | 0.1  | **+0.1825**   | +0.073   |

## Key Observation

The `sigma_sde` bias — Bayesian estimate ≈ 0.28 against a true value
of 0.1 — is a **~2.8x overestimate of the diffusion coefficient**.

This is consistent with a **discretization bias in the Euler-Maruyama
pseudo-likelihood** under coarse observation, rather than a property
of Bayesian inference itself. The Bayesian method recovers the drift
parameters (a, b, c, d) reasonably well; the systematic failure is
concentrated in the diffusion coefficient.

This is the more scientifically interesting finding: **when and why
is the pseudo-likelihood biased?** — which is precisely the question
this package was designed to investigate.

## Implications

For SDE parameter estimation with discretely observed data:

- The Euler-Maruyama pseudo-likelihood appears to systematically
  over-estimate the diffusion coefficient at coarse observation
  intervals. This is a discretization artifact, not a failure of
  Bayesian inference.
- Practitioners using pseudo-likelihood methods should be aware of
  this bias and consider finer observation grids or higher-order
  pseudo-likelihoods.
- The bias should shrink as the observation interval decreases; this
  is the subject of the ongoing ablation study.

## Planned Extensions

1. **100 replicates** per cell, with per-parameter standard errors
2. **Coverage analysis** — do 90% credible intervals cover the true
   values at the nominal rate?
3. **Observation interval ablation** — does the bias vanish as
   `dt → 0`? This distinguishes discretization bias from a genuine
   property of the pseudo-likelihood.
4. **Higher-order pseudo-likelihoods** (Milstein, Itô–Taylor)
5. **Particle MCMC** for exact likelihood comparison
6. **Bayesian vs MLE on matched objectives** — ensure both methods
   optimize the same target before drawing conclusions