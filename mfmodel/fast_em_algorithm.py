from scipy.linalg import block_diag
import scipy
import mlrfit as mf
import numpy as np


from mfmodel.utils import *
from mfmodel.inverse import *




def get_sparse_F_si_col_sparsity(F_compressed, ranks, F_hpart, group):
    res = np.zeros(F_compressed.shape)
    for level, gi in enumerate(group):
        r1, r2 = F_hpart["lk"][level][gi], F_hpart["lk"][level][gi+1]
        res[r1:r2, ranks[:level].sum() : ranks[:level+1].sum()] = F_compressed[r1:r2, ranks[:level].sum() : ranks[:level+1].sum()]
    return res


def fast_EM_intermediate_matrices(F_Lm1, D, F_hpart, Y, ranks, si, si_groups, row_selectors):
    r1, r2 = row_selectors[si: si+2]
    N, n = Y.shape
    tilde_F_ci = get_sparse_F_si_col_sparsity(F_Lm1, ranks, F_hpart, si_groups[si]) # n x (r-1)
    assert tilde_F_ci.shape == (n, ranks[:-1].sum())
    Sigma0_inv_F_ci = np.zeros(tilde_F_ci.shape)
    for j in range(tilde_F_ci.shape[1]):
        Sigma0_inv_F_ci[:, j:j+1] = iterative_refinement(ranks, tilde_F_ci[:, j:j+1], F_Lm1, D, F_hpart, 
                                                         eps=1e-12, max_iter=20, printing=False)
    F_ciT_Sigma0_inv_F_ci = tilde_F_ci.T @ Sigma0_inv_F_ci # (r-1) x (r-1)
    del tilde_F_ci
    Y_Sigma0_inv_F_ci = Y @ Sigma0_inv_F_ci # N x (r-1)
    ci_B_ci = N * (np.eye(ranks[:-1].sum()) - F_ciT_Sigma0_inv_F_ci) + Y_Sigma0_inv_F_ci.T @ Y_Sigma0_inv_F_ci
    del F_ciT_Sigma0_inv_F_ci
    ri_At_ci_t = Y_Sigma0_inv_F_ci.T @ Y[:, r1:r2] # (r-1) x (r2-r1)
    return ri_At_ci_t, ci_B_ci, r1, r2


def fast_EM_get_F(F0, D0, Y, ranks, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    F1 = np.zeros(F0.shape)
    num_sparsities = row_selectors.size - 1
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = fast_EM_intermediate_matrices(F0, D0, F_hpart, Y, ranks, si, si_groups, row_selectors)
        F1[r1:r2, :] = np.linalg.solve(ci_B_ci, ri_At_ci_t).T
    return F1


def fast_EM_get_D(F0, D0, F1, Y, ranks, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    N, n = Y.shape
    num_sparsities = row_selectors.size - 1
    D1 = np.zeros(n)
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = fast_EM_intermediate_matrices(F0, D0, F_hpart, Y, ranks, si, si_groups, row_selectors)
        ri_F1_ciT = F1[r1:r2]
        D1[r1:r2] = (1/N) * ( np.einsum('ij,ji->i', Y[:, r1:r2].T, Y[:, r1:r2]) \
                            - 2 * np.einsum('ij,ji->i', ri_F1_ciT, ri_At_ci_t) \
                            + np.einsum('ij,jk,ki->i', ri_F1_ciT, ci_B_ci, ri_F1_ciT.T))
    return D1


def fast_em_algorithm(Y, F0, D0, F_hpart, row_selectors, si_groups, ranks, lls=False, max_iter=200, 
                        eps=1e-12, freq=50, printing=False):
        """
            Fast EM algorithm
            Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
        """
        loglikelihoods = [-np.inf]
        obj = loglikelihoods[-1]
        N = Y.shape[0]
        for t in range(max_iter):
                if lls:
                    obj = fast_loglikelihood_value(F0, D0, Y, ranks, F_hpart)
                    loglikelihoods += [obj]
                if printing and t % freq == 0:
                    print(f"{t=}, {obj=}")
                F1 = fast_EM_get_F(F0, D0, Y, ranks, F_hpart, row_selectors, si_groups)
                D1 = fast_EM_get_D(F0, D0, F1, Y, ranks, F_hpart, row_selectors, si_groups)
                F0, D0 = F1, D1
                assert D1.min() >= -1e-8 and loglikelihoods[-2] - 1e-8 <= loglikelihoods[-1]
                if loglikelihoods[-1] - loglikelihoods[-2] < eps * abs(loglikelihoods[-2]):
                    print(f"terminating at {t=}")
                    break
        if printing: print(f"{t=}, {obj=}")
        return F0, D0, loglikelihoods
        

def perm_hat_Sigma(F:np.ndarray, D:np.ndarray, hpart:mf.EntryHpartDict, ranks:np.ndarray):
        """
        Compute permuted hat_A with each Sigma_level being block diagonal matrix 
        """
        num_levels = ranks.size
        perm_hat_Sigma = np.diag(D)
        for level in range(num_levels - 1):
            Sigma_level = []
            num_blocks = len(hpart['lk'][level])-1
            for block in range(num_blocks):
                r1, r2 = hpart['lk'][level][block], hpart['lk'][level][block+1]
                Sigma_level += [ F[:,ranks[:level].sum() : ranks[:level+1].sum()][r1:r2] @ F[:,ranks[:level].sum() : ranks[:level+1].sum()][r1:r2].T ]
            perm_hat_Sigma += block_diag(*Sigma_level)
        return perm_hat_Sigma


def fast_exp_true_loglikelihood_value(F0, D0, ranks, F_hpart, num_factors, tol1=1e-8, tol2=1e-8):
    """
        Expected log-likelihood under the true model
    """
    n = F0.shape[0]
    logdet = fast_logdet_FFtpD(F0, D0, ranks, F_hpart, num_factors, tol1=tol1, tol2=tol2)    

    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * logdet - n/2
    return obj


def fast_logdet_FFtpD(F0, D0, ranks, F_hpart, num_factors, tol1=1e-8, tol2=1e-8):
    rescaled_sparse_F = mf.convert_compressed_to_sparse(np.power(D0, -0.5)[:, np.newaxis] * F0, 
                                             F_hpart, 
                                             ranks[:-1])
    try:
        sigmas = scipy.sparse.linalg.svds( rescaled_sparse_F, k=num_factors//2, return_singular_vectors=False, which='LM', tol=tol1)
    except:
        sigmas = scipy.sparse.linalg.svds( rescaled_sparse_F, k=num_factors//2, return_singular_vectors=False, which='LM', tol=1e-5)
    try:
        last_sigma = scipy.sparse.linalg.svds(rescaled_sparse_F, k=num_factors-num_factors//2, return_singular_vectors=False, which='SM', tol=tol2)
    except:
        last_sigma = scipy.sparse.linalg.svds(rescaled_sparse_F, k=num_factors-num_factors//2, return_singular_vectors=False, which='SM', tol=1e-5)
    assert last_sigma.size + sigmas.size == num_factors and num_factors == rescaled_sparse_F.shape[1]
    del rescaled_sparse_F
    
    logdet = np.log(D0).sum() + np.log(sigmas**2 + 1).sum() + np.log(last_sigma**2 + 1).sum()
    return logdet


def fast_loglikelihood_value(F0, D0, Y, ranks, F_hpart, num_factors, tol1=1e-7, tol2=1e-7):
    """
        Average log-likelihood of observed data
    """
    N, n = Y.shape
    logdet = fast_logdet_FFtpD(F0, D0, ranks, F_hpart, num_factors, tol1=tol1, tol2=tol2)  

    Sigma_inv_Yt = np.zeros(Y.T.shape)
    for j in range(N):
        Sigma_inv_Yt[:, j:j+1] = iterative_refinement(ranks, Y.T[:, j:j+1], F0, D0, F_hpart, 
                                                         eps=1e-12, max_iter=20, printing=False)
    # trace(Sigma^{-1}Y^T Y)
    tr_Sigma_inv_YtY = np.einsum('ij,ji->i', Y, Sigma_inv_Yt).sum()
    obj = - (N*n/2) * np.log(2 * np.pi) - (N/2) * (logdet) - 0.5 * tr_Sigma_inv_YtY 
    return obj / N
