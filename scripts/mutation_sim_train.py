import numpy as np
from scipy.stats import poisson

import main2_fixed_denovo
np.random.seed(100)


def main(M, n_signatures, lambd):
    M = M.to_numpy()
    n_samples, n_mutations = M.shape
    O = np.ones((n_samples, n_mutations), dtype=int)
    E = np.full((n_samples, n_signatures), 0.00001)
    S = np.random.rand(n_signatures*n_mutations).reshape(n_signatures, n_mutations)
    topt = np.float64("Inf")
    tedge = np.float64("Inf")
    # lambd = 0
    if( np.any(E< 0)):
        E = np.maximum(E,0)
    pmf_e = []
    pmf_s = []
    d_mse_e = []
    d_mse_s = []
    mse_old = np.inf
    pmf_old = np.inf
    for _ in range(5):
        # # print("Step is script:", _ )
        E = main2_fixed_denovo.running_simulation_new(E, M, S, O, topt, tedge, lambd)# lambd)
        S = main2_fixed_denovo.running_simulation_new(S.T, M.T, E.T, O.T, topt, tedge, 0).T
        mse_e = main2_fixed_denovo.Frobinous(M,S,E,O)
        d_mse_s.append(mse_e)
        loss = -poisson.logpmf(M,(E@S)*O)
        pmf_s.append(np.mean(loss))
        # print("MSE of S", mse_e)
        conv = main2_fixed_denovo.convergence(mse_e, mse_old)
        conv = main2_fixed_denovo.convergence(np.mean(loss), pmf_old)
        # print("LOSS PMF E", np.mean(loss))
        # print("LOSS S", np.mean(loss))
    #    if (conv==True):
    #        # print("The pmf converge")
    #        break
        mse_old = mse_e
        pmf_old = np.mean(loss)
    
    
    if(np.any(E < 0)):
        E = np.maximum(E, 0)
    E /= E.sum(axis=-1, keepdims= True)
    E[np.isnan(E)] = 0
    if(np.any(S < 0)):
        S = np.maximum(S, 0)
    S /= S.sum(axis=-1, keepdims= True)
    S[np.isnan(S)] = 0
    return np.array(E), np.array(S)
