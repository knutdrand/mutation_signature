"""Main module."""
import numpy as np
from scipy.stats import poisson
import main2_fixed_denovo


def main(M, S):
    M = M.to_numpy()
    S = S.to_numpy()
    n_samples, n_mutations = M.shape
    n_signatures = S.shape[0]
    O = np.ones((n_samples, n_mutations), dtype=int)
    E_true = np.abs(np.random.laplace(loc=0, scale=1, size=n_samples*n_signatures).reshape(n_samples, n_signatures))
    E = np.full_like(E_true, 0.00001)

    topt = np.float64("Inf")
    tedge = np.float64("Inf")
    if(np.any(E < 0)):
        E = np.maximum(E, 0)
    E = main2_fixed_denovo.running_simulation_new(E, M, S, O, topt, tedge, 0)
    loss = -poisson.logpmf(M, (E@S)*O)
    return E, np.mean(loss)
