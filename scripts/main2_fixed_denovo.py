import numpy as np
from scipy.stats import poisson


def compute_local_gradients(E, M, S, O):
    n_samples, n_signatures, n_mutations = (E.shape[0], S.shape[0], M.shape[1])
    local_gradients = np.empty_like(E)

    for i in range(n_samples):
        for r in range(n_signatures):
            # print(type(M[i]))
            # print(type(S[r]))
            # print(M.dtype, S.dtype)
            numerator = M[i]*S[r]
            denumerator_sum = np.array([E[i] @ S[:, k] for k in range(n_mutations)])
            denumerator_sum_c = denumerator_sum + 0.000001
            local_gradients[i, r] = np.sum((numerator/denumerator_sum_c) - O[i]*S[r])

    return local_gradients


def compute_hessians(E, M, S):
    denominatior = (E@S)**2 + 0.000001
    numerator = M[:, None, None, :]*S[None, :, None, :]*S[None, None, :, :]
    res = numerator/denominatior[:, None, None, :]
    return -res.sum(axis=-1)
    # 
    # 
    # denominator = np.einsum(E, S, "ir,rk->ik"
    # 
    # for i in range(n_samples):
    #     hessian = np.empty((n_signatures, n_signatures))
    #     for r in range(n_signatures):
    #         for s in range(n_signatures):
    #             numerator = M[i]*S[r]*S[s]
    #             denominatior = np.array([(E[i] @ S[:, k])**2
    #                                      for k in range(n_mutations)])
    #             hessian[r, s] = np.sum(-(numerator/denominatior))
    #             # # print(hessian.shape)
    #     hessians.append(hessian)
    # 
    # return hessians


def compute_global_gradient(E, local_gradients, lambd):
    cond_a = local_gradients-lambd*np.sign(E)
    cond_b = local_gradients-lambd*np.sign(local_gradients)
    cond_c = 0
#    # print(np.where(E!=0, cond_a, np.where(np.abs(local_gradients)>lambd, cond_b, cond_c)))
    return np.where(E != 0, cond_a, np.where(np.abs(local_gradients) > lambd, cond_b, cond_c))


def compute_topt(E, local_gradients, global_gradients, hessians):
    numerator = np.sum(global_gradients * local_gradients)
    gg_vectors = (gg[:, None] for gg in global_gradients)
    denominatior = sum([gg.T @ hessians @ gg for gg, hessians in zip(gg_vectors, hessians)])
    topt = - (numerator/denominatior)
    return topt


def compute_t_edge_ori(E, global_gradients):
    global_gradients = global_gradients.flatten()
    E = E.flatten()
    mask = np.sign(E) == -np.sign(global_gradients)
    mask &= (np.sign(E) != 0)
    if not np.any(mask):
        return np.inf
    return np.min(-(E/global_gradients)[mask])

"Tedge without NAN"

def compute_t_edge_nan(E, global_gradients):
    global_gradients = global_gradients.flatten()
    E = E.flatten()
    mask = np.sign(E) == -np.sign(global_gradients)
    mask &= (np.sign(E) != 0)
    if not np.any(mask):
        return np.inf
    E_gradient = -(E/global_gradients)[mask]
    return np.min(E_gradient[~np.isnan(E_gradient)])

"Remove gradient with zero"

def compute_t_edge(E, global_gradients):
    global_gradients = global_gradients.flatten()
    E = E.flatten()
    ind = np.where(global_gradients == 0)
    E_Conv = np.delete(E, ind[0])
    global_gradients_conv = np.delete(global_gradients, ind[0])
    mask = np.sign(E_Conv) == -np.sign(global_gradients_conv)
    mask &= (np.sign(E_Conv) != 0)
    if not np.any(mask):
        return np.inf
    assert np.all(global_gradients_conv != 0)
    return np.min(-(E_Conv/global_gradients_conv)[mask])


def min_topt_tedge(topt, tedge):
    topt_tedge = np.minimum(float(topt), float(tedge))
    return topt_tedge


def update_exposure_gradient(E, global_gradients, topt_tedge):
    if(np.any(E < 0)):
        E = np.maximum(E, 0)
    return E + topt_tedge * global_gradients


def newton_raphson1(E, global_gradients, hessians):
    nr = []
    v1 = []
    H = hessians
    active_mask = (E != 0)
    assert np.all(E >= 0)
    active_hessians = []
    for E_row, gradient_row, hessian in zip(E, global_gradients, H):
        non_active = ((E_row == 0) | (gradient_row == 0))
        active_mask = ~non_active
        # active_mask = (gradient_row != 0) & (~((E_row == 0) & (gradient_row > 0)))
        active_gradients = gradient_row[active_mask]
        active_hessian = hessian[active_mask][:, active_mask]
        new_row = E_row.copy()
        det = np.linalg.det(active_hessian)
        active_hessians.append(det**-1)
        if det == 0:
            assert False
        if det < 10e-10:
            return None
        # # print("#########", np.linalg.inv(active_hessian))
        new_row[active_mask] = E_row[active_mask] - np.linalg.inv(active_hessian) @ active_gradients
        nr.append(new_row)
    # print("H", np.mean(active_hessians))
    v1 = np.array(nr)
    return v1


def update_exposure_NR(E, global_gradients, topt, tedge, new_E):
    if(np.any(E < 0)):
        E = np.maximum(E, 0)
    assert topt <= tedge
    return np.where(np.sign(new_E) == np.sign(E),
                    new_E,
                    E + topt * global_gradients)


def check(global_gradients):
    counter = 0
    global_gradients = global_gradients.flatten()
    for gradient in global_gradients:
        if gradient != 0:
            counter += 1
    if counter == 0:
        return(True)
    else:
        return(False)


def convergence(E, E_hat, tol=0.00000001):
    conv = []
    conv = np.all(np.abs( E_hat - E )   < tol)
    # if (conv==True):
    #     print ("The convergence value is:",conv)
    return conv


def mean_exposure(E):
    m = []
    m = np.mean(E)
    return m

def Frobinous(M, S, E, O):
    from numpy import linalg as LA
    fibo = []
    fibo1 = []
 #   # print("The shape of E is:",E.shape)
 #   # print("The shape oSf S is:",S.shape)
    fibo1 =  (E @ S)*O
    # # print(fibo1)
    fibo = LA.norm(M - fibo1, ord = 2)
    return fibo


def running_simulation_new(E, M, S, O, topt, tedge, lambd):
    # global_gradients = np.full((21, 30), 1)
    ###n_normal, n_newton = (0, 0)
    # loss_func = lambda e: -np.mean(poisson.logpmf(M, (e@S)*O))
    # loss = -poisson.logpmf(M,(E@S)*O)
    old_loss = np.inf
    pmf_s = []
    for step in range(50):
        ## print("Step is:", step)
        E_hat = E
        if(np.any(E < 0)):
        #    # print("############", np.min(E))
            E = np.maximum(E, 0)
        #    # print("############", np.min(E))
        local_gradients = compute_local_gradients(E, M, S, O)
        hessians = compute_hessians(E, M, S)
        global_gradients = compute_global_gradient(E, local_gradients, lambd)
        topt = compute_topt(E, local_gradients, global_gradients, hessians)
        tedge = compute_t_edge(E, global_gradients)
        minimun_topt_tedge = min_topt_tedge(topt, tedge)
        #assert not np.any(np.isnan(E)) and not np.any(np.isinf(E))
        #assert not np.any(np.isnan(old_loss))
        if topt >= tedge:
            if(np.any(E < 0)):
                E = np.maximum(E, 0)
            E = update_exposure_gradient(E, global_gradients, minimun_topt_tedge)
            # l = loss_func(E)
            # old_loss = l
            ## print("Normal", loss_func(E), np.mean(global_gradients**2), topt, tedge)
        else:
            if(np.any(E < 0)):
                E = np.maximum(E, 0)
            newton_raphason = newton_raphson1(E, global_gradients, hessians)
            if newton_raphason is None:
                if(np.any(E < 0)):
                    E = np.maximum(E, 0)
                E = update_exposure_gradient(E, global_gradients, minimun_topt_tedge)
            else:
                if(np.any(E < 0)):
                    E = np.maximum(E, 0)
                E = update_exposure_NR(E, global_gradients, topt, tedge, newton_raphason)
     #       conv = convergence(E, E_hat)
     #        l = loss_func(E)
     #        # print("LOSSS", l)
     #        # print("OLD_LOSS", old_loss)
     #        loss = -poisson.logpmf(M,(E@S)*O)
     #        conv = convergence(old_loss, l)
     #        if (conv==True):
     #           break
     #           # print("The pmf converge")
               #exit()
            #     # print(E.mean())
            #     # print(global_gradients.mean())
            #     # print(minimun_topt_tedge, minimun_topt_tedge)
            # old_loss = l
            # # print("Newton", loss_func(E), np.mean(global_gradients**2), topt, tedge)

        # topt = compute_topt(E, local_gradients, global_gradients, hessians)
        # tedge =compute_t_edge(E, global_gradients)
        # local_gradients = compute_local_gradients(E, M, S, O)
        # hessians = compute_hessians(E, M, S, O)
        # global_gradients= compute_global_gradient(E, local_gradients, lambd= 0)
    #### print(n_normal, n_newton)
    if(np.any(E < 0)):
        E = np.maximum(E, 0)
    #     # print("OLD LOSS_MAIN",old_loss )
    #     loss = -poisson.logpmf(M,(E@S)*O)
    #     pmf_s.append(np.mean(loss))
    #     conv = convergence(old_loss, np.mean(loss))
    #     old_loss = np.mean(loss)
    #     # print("LOSS_MAIN", np.mean(loss))
    #     if (conv==True):
    #         # print("The pmf converge_MAIN")
    #         # print("PMF FINAL",np.mean(pmf_s))
    #         break
    return E

## Newton-Raphson

##Backup
"""
"""
#def Frobinous(M, S, E, O):
#    from numpy import linalg as LA
#    fibo = []
#    fibo1 = []
 #   # print("The shape of E is:",E.shape)
 #   # print("The shape oSf S is:",S.shape)
 #   fibo1 =  (E @ S)*O
    # # print(fibo1)
 #   fibo = LA.norm(M - fibo1, ord = 2)
 #   return fibo
def mse(E, E_hat):
    mse_error = []
    # from sklearn.metrics import mean_squared_error
    mse_error = np.square(np.subtract(E, E_hat)).mean()
    return mse_error

def cosine_similarity(E, cosmic):
    from numpy.linalg import norm
    cosine = []
    cosine =np.dot(E,cosmic)/(norm(E) * norm(cosmic))
    return cosine



