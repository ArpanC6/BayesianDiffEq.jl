\# SDE Bias Study Results



\## Setup



\- 10 independent trials

\- Stochastic Lotka-Volterra SDE

\- True parameters: \[1.5, 1.0, 3.0, 1.0, 0.1]

\- Bayesian: Turing.jl + Euler-Maruyama pseudo-likelihood

\- MLE: Optimization.jl + Nelder-Mead



\## Results



\### Mean Bias (across 10 trials)



| Parameter | True | Bayesian Bias | MLE Bias | Winner |

|-----------|------|---------------|----------|--------|

| a | 1.5 | 0.2053 | 0.0302 | MLE |

| b | 1.0 | 0.0764 | 0.0333 | MLE |

| c | 3.0 | -0.2923 | -0.137 | MLE |

| d | 1.0 | 0.0663 | 0.0046 | MLE |

| sigma | 0.1 | 0.1825 | 0.073 | MLE |



\### Key Finding



MLE consistently recovers parameters more accurately than Bayesian 

pseudo-likelihood. The Bayesian bias is 2-6x larger than MLE bias 

across all parameters.



This suggests either:

1\. The Euler-Maruyama approximation is insufficient

2\. NUTS has divergences (warnings observed)

3\. The pseudo-likelihood itself is biased



\## Implications



For SDE parameter estimation:

\- MLE is more accurate but gives no uncertainty

\- Bayesian gives full posterior but with bias

\- Trade-off depends on application



\## Future Work



1\. Higher-order pseudo-likelihood (Milstein)

2\. Particle MCMC for exact likelihood

3\. Comparison with R pCODE

4\. Larger sample sizes

