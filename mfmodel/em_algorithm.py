from scipy.linalg import block_diag
import mlrfit as mf
import numpy as np

from tqdm import tqdm
import scipy


from mfmodel.utils import *


def block_diag_FFt(level:int, hpart_entry:mf.EntryHpartDict, F_level:np.ndarray):
    A_level = []
    num_blocks = len(hpart_entry['lk'][level])-1
    for block in range(num_blocks):
        r1, r2 = hpart_entry['lk'][level][block], hpart_entry['lk'][level][block+1]
        A_level += [ F_level[r1:r2] @ F_level[r1:r2].T ]
    return block_diag(*A_level)
    

def perm_hat_Sigma(F_compressed:np.ndarray, D:np.ndarray, hpart_entry:mf.EntryHpartDict, ranks:np.ndarray):
    """
        Compute permuted hat_Sigma with each A_level being block diagonal matrix 
    """
    num_levels = ranks.size
    assert F_compressed.shape == (D.shape[0], ranks[:-1].sum())
    perm_hat_A = np.copy(np.diag(D.flatten())) + 0
    for level in range(num_levels - 1):
        perm_hat_A += block_diag_FFt(level, hpart_entry, F_compressed[:,ranks[:level].sum():ranks[:level+1].sum()])
    return perm_hat_A


def perm_tildeF_tildeFt(F_compressed:np.ndarray, hpart_entry:mf.EntryHpartDict, ranks:np.ndarray):
    """
        Compute permuted \tilde F \tilde F^T with each A_level being block diagonal matrix 
    """
    num_levels = ranks.size
    assert F_compressed.shape[1] == ranks[:-1].sum()
    perm_hat_A = np.zeros((F_compressed.shape[0], F_compressed.shape[0]))
    for level in range(num_levels - 1):
        perm_hat_A += block_diag_FFt(level, hpart_entry, F_compressed[:,ranks[:level].sum():ranks[:level+1].sum()])
    return perm_hat_A


def perm_hat_Sigma_sp(sparse_F:np.ndarray, D:np.ndarray):
    """
        Compute permuted hat_Sigma with each A_level being block diagonal matrix 
    """
    return sparse_F @ sparse_F.T + np.diag(D.flatten())


def EM_intermediate_matrices(tilde_F, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors):
    r1, r2 = row_selectors[si: si+2]
    N, n = Y.shape
    si_col = group_to_indices(si_groups[si], part_sizes, ranks)
    tilde_F_ci = tilde_F[:, si_col].toarray()
    Sigma0_inv_F_ci = scipy.linalg.lu_solve((lu, piv), tilde_F_ci)
    F_ciT_Sigma0_inv_F_ci = tilde_F_ci.T @ Sigma0_inv_F_ci
    del tilde_F_ci
    Y_Sigma0_inv_F_ci = Y @ Sigma0_inv_F_ci
    ci_B_ci = N * (np.eye(si_col.size) - F_ciT_Sigma0_inv_F_ci) + Y_Sigma0_inv_F_ci.T @ Y_Sigma0_inv_F_ci
    ri_At_ci_t = Y_Sigma0_inv_F_ci.T @ Y[:, r1:r2]
    return ri_At_ci_t, ci_B_ci, r1, r2


def exp_true_loglikelihood_value(Sigma):
    """
        Expected log-likelihood under the true model
    """
    n = Sigma.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(Sigma)
    assert sign == 1, print(sign, logabsdet, np.linalg.det(Sigma))
    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * (logabsdet) - n/2
    return obj


def exp_loglikelihood_value(true_Sigma, fitted_Sigma, fitted_lu, fitted_piv):
    """
        Expected log-likelihood under the true model
    """
    n = true_Sigma.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(fitted_Sigma)
    assert sign == 1, print(sign, logabsdet, np.linalg.det(fitted_Sigma))
    fitted_Sigma_true_Sigma = scipy.linalg.lu_solve((fitted_lu, fitted_piv), true_Sigma)
    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * (logabsdet) - (1/2) * np.trace(fitted_Sigma_true_Sigma)
    return obj


def loglikelihood_value(Sigma, lu, piv, Y):
    """
        Average log-likelihood of observed data
    """
    N, n = Y.shape
    (sign, logabsdet) = np.linalg.slogdet(Sigma)
    assert sign == 1, print(sign, logabsdet, np.linalg.det(Sigma))
    Sigma_inv_Yt = scipy.linalg.lu_solve((lu, piv), Y.T)
    # trace(Sigma^{-1}Y^T Y)
    tr_Sigma_inv_YtY = np.einsum('ij,ji->i', Y, Sigma_inv_Yt).sum()
    obj = - (N*n/2) * np.log(2 * np.pi) - (N/2) * (logabsdet) - 0.5 * tr_Sigma_inv_YtY
    # obj = - (N*n/2) * np.log(2 * np.pi) - (N/2) * fast_logdet(Sigma) - 0.5 * tr_Sigma_inv_YtY
    return obj / N


def EM_get_F(F0, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    F1 = np.zeros(F0.shape)
    num_sparsities = row_selectors.size - 1
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        F1[r1:r2, :] = np.linalg.solve(ci_B_ci, ri_At_ci_t).T
    del tilde_F0
    return F1


def EM_get_D(F0, F1, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    N, n = Y.shape
    num_sparsities = row_selectors.size - 1
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    D1 = np.zeros(n)
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        ri_F1_ciT = F1[r1:r2]
        D1[r1:r2] = (1/N) * ( np.einsum('ij,ji->i', Y[:, r1:r2].T, Y[:, r1:r2]) \
                            - 2 * np.einsum('ij,ji->i', ri_F1_ciT, ri_At_ci_t) \
                            + np.einsum('ij,jk,ki->i', ri_F1_ciT, ci_B_ci, ri_F1_ciT.T))
    return D1


def em_algorithm(n, Y, part_sizes, F_hpart, row_selectors, si_groups, ranks, max_iter=200, 
                 eps=1e-12, freq=50, printing=False, F0=None, D0=None):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    loglikelihoods = [-np.inf]
    N = Y.shape[0]
    if F0 is None:
        # F0 = np.random.randn(n, ranks[:-1].sum()) * 0.015
        # _, C = low_rank_approx(Y, dim=ranks[0], symm=False)
        # F0[:, :ranks[0]] = C / np.sqrt(N)
        _, C = low_rank_approx(Y, dim=ranks[:-1].sum(), symm=False)
        F0 = C / np.sqrt(N)
        D0 = np.maximum(np.einsum('ij,ji->i', Y.T, Y) / N - np.diag(perm_tildeF_tildeFt(F0, F_hpart, ranks)), 1e-4)
        # D0 = np.maximum(np.einsum('ij,ji->i', Y.T, Y) / N - np.einsum('ij,ji->i', F0, F0.T), 1e-8)
        # F0, D0 = np.random.randn(n, ranks[:-1].sum()), np.square(np.random.rand(n)) + 1e-6
    assert D0.shape == (n, ) and F0.shape == (n, ranks[:-1].sum())
    for t in range(max_iter):
            Sigma0 = perm_hat_Sigma(F0, D0, F_hpart, ranks)
            lu, piv = scipy.linalg.lu_factor(Sigma0)
            obj = loglikelihood_value(Sigma0, lu, piv, Y)
            loglikelihoods += [obj]
            if printing and t % freq == 0:
                print(f"{t=}, {obj=}")
            F1 = EM_get_F(F0, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups)
            D1 = EM_get_D(F0, F1, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups)
            F0, D0 = F1, D1
            assert D1.min() >= -1e-8 and loglikelihoods[-2] - 1e-8 <= loglikelihoods[-1]
            if loglikelihoods[-1] - loglikelihoods[-2] < eps * abs(loglikelihoods[-2]):
                print(f"terminating at {t=}")
                break
    if printing:
        print(f"{t=}, {obj=}")
    return loglikelihoods, F0, D0


def q_thetap_theta_value(F1, D1, F0, lu, piv, Y, row_selectors, si_groups, ranks, part_sizes, F_hpart):
    n, s = F1.shape
    N, n = Y.shape
    assert F1.shape[0] == Y.shape[1]
    num_sparsities = row_selectors.size - 1
    obj = - (N*(n+s)/2) * np.log(2 * np.pi) - (N/2) * (np.log(D1)).sum()
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    # assert D.shape == (D.size, )
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        ri_F1_ciT = F1[r1:r2, :]
        M = np.einsum('ij,ji->i', Y[:, r1:r2].T, Y[:, r1:r2]) \
                        - 2 * np.einsum('ij,ji->i', ri_F1_ciT, ri_At_ci_t) \
                        + np.einsum('ij,jk,ki->i', ri_F1_ciT, ci_B_ci, ri_F1_ciT.T)
        obj += - 0.5 * np.sum(M * np.power(D1[r1:r2], -1)) - 0.5 * np.trace(ci_B_ci)
    return obj / N


def frob_fit_loglikehood(undermuted_A, Y, F_hpart, hpart, ranks, printing=True, eps_ff=1e-3):
    hat_A = mf.MLRMatrix()
    hat_A.hpart = hpart
    losses = hat_A.factor_fit(undermuted_A, ranks, hat_A.hpart, eps_ff=eps_ff, PSD=True, freq=1, \
                                    printing=printing, max_iters_ff=50, symm=True)
    F_frob, D_frob = hat_A.B[:, :-1], np.square(hat_A.B[:, -1])
    Sigma_frob = perm_hat_Sigma(F_frob, D_frob, F_hpart, ranks)
    lu, piv = scipy.linalg.lu_factor(Sigma_frob)
    obj_frob = loglikelihood_value(Sigma_frob, lu, piv, Y)
    print(f"FR: {obj_frob = }")
    print(f"{mf.rel_diff(hat_A.matrix(), den=undermuted_A)=} \n{np.linalg.slogdet(hat_A.matrix())=}")
    return obj_frob
