from scipy.linalg import block_diag
import torch
import mlrfit as mf
import numpy as np

from tqdm import tqdm
import scipy
from sklearn.utils.extmath import fast_logdet


from mfmodel.utils import *
from mfmodel.inverse import *





def fast_EM_intermediate_matrices(tilde_F, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors):
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


def fast_loglikelihood_value(Sigma, lu, piv, Y):
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


def fast_EM_get_F(F0, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    F1 = np.zeros(F0.shape)
    num_sparsities = row_selectors.size - 1
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = fast_EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        F1[r1:r2, :] = np.linalg.solve(ci_B_ci, ri_At_ci_t).T
    del tilde_F0
    return F1


def fast_EM_get_D(F0, F1, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    N, n = Y.shape
    num_sparsities = row_selectors.size - 1
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    D1 = np.zeros(n)
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = fast_EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        ri_F1_ciT = F1[r1:r2]
        D1[r1:r2] = (1/N) * ( np.einsum('ij,ji->i', Y[:, r1:r2].T, Y[:, r1:r2]) \
                            - 2 * np.einsum('ij,ji->i', ri_F1_ciT, ri_At_ci_t) \
                            + np.einsum('ij,jk,ki->i', ri_F1_ciT, ci_B_ci, ri_F1_ciT.T))
    return D1


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

