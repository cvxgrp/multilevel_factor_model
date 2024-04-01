from scipy.linalg import block_diag
import torch
import mlrfit as mf
import numpy as np

from tqdm import tqdm
import scipy
from sklearn.utils.extmath import fast_logdet


def block_diag_FFt(level:int, hpartentry:mf.EntryHpartDict, F_level:np.ndarray):
    A_level = []
    num_blocks = len(hpartentry['lk'][level])-1
    for block in range(num_blocks):
        r1, r2 = hpartentry['lk'][level][block], hpartentry['lk'][level][block+1]
        A_level += [ F_level[r1:r2] @ F_level[r1:r2].T ]
    if torch.is_tensor(F_level):
        return torch.block_diag(*A_level)
    else:
        return block_diag(*A_level)
    

def perm_hat_Sigma(F:np.ndarray, D:np.ndarray, hpartentry:mf.EntryHpartDict, ranks:np.ndarray):
    """
    Compute permuted hat_Sigma with each A_level being block diagonal matrix 
    """
    num_levels = ranks.size
    perm_hat_A = np.copy(np.diag(D.flatten()))
    for level in range(num_levels - 1):
        perm_hat_A += block_diag_FFt(level, hpartentry, F[:,ranks[:level].sum():ranks[:level+1].sum()])
    return perm_hat_A


def group_to_indices(group, part_sizes, ranks):
    cumsum = 0
    indices = []
    for level, gi in enumerate(group):
        indices += [np.arange(cumsum + gi * ranks[level], cumsum + (gi + 1) * ranks[level])]
        cumsum += ranks[level] * part_sizes[level]
    indices = np.concatenate(indices, axis=0)
    assert indices.size == ranks[:-1].sum()
    return indices


def EM_intermediate_matrices(tilde_F, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors):
    r1, r2 = row_selectors[si: si+2]
    N, n = Y.shape
    si_col = group_to_indices(si_groups[si], part_sizes, ranks)
    tilde_F_ci = tilde_F[:, si_col].todense()
    Sigma0_inv_F_ci = scipy.linalg.lu_solve((lu, piv), tilde_F_ci)
    F_ciT_Sigma0_inv_F_ci = tilde_F_ci.T @ Sigma0_inv_F_ci
    Y_Sigma0_inv_F_ci = Y @ Sigma0_inv_F_ci
    ci_B_ci = N * (np.eye(si_col.size) - F_ciT_Sigma0_inv_F_ci) + Y_Sigma0_inv_F_ci.T @ Y_Sigma0_inv_F_ci
    ri_At_ci_t = Y_Sigma0_inv_F_ci.T @ Y[:, r1:r2]
    return ri_At_ci_t, ci_B_ci, r1, r2


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


def loglikelihood_value(Sigma, lu, piv, Y):
    """
    Average log-likelihood of observed data
    """
    N, n = Y.shape
    # obj = - (N*n/2) * np.log(2 * np.pi) - (N/2) * (np.log(np.linalg.det(Sigma)))
    obj = - (N*n/2) * np.log(2 * np.pi) - (N/2) * fast_logdet(Sigma)
    # trace(Sigma^{-1}Y^T Y)
    Sigma_inv_Yt = scipy.linalg.lu_solve((lu, piv), Y.T)
    M = np.einsum('ij,ji->i', Y, Sigma_inv_Yt)
    obj += - 0.5 * np.sum(M)
    return obj / N


def EM_get_F(F0, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    F1 = np.zeros(F0.shape)
    num_sparsities = row_selectors.size - 1
    tilde_F0 = mf.convert_compressed_to_sparse(F0, F_hpart, ranks[:-1])
    # for si in tqdm(range(num_sparsities)):
    for si in range(num_sparsities):
        ri_At_ci_t, ci_B_ci, r1, r2 = EM_intermediate_matrices(tilde_F0, Y, lu, piv, ranks, si, part_sizes, si_groups, row_selectors)
        F1[r1:r2, :] = np.linalg.solve(ci_B_ci, ri_At_ci_t).T
    return F1


def EM_get_D(F0, F1, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups):
    # get D1
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


def row_col_selections(hpart):
    """
        Define row and col selectors for each row sparsity pattern of F
    """
    num_levels = len(hpart['rows']['lk'])
    L = num_levels - 1
    num_sparsities = len(hpart['rows']['lk'][L - 1]) - 1
    print(f"{num_levels=}, {num_sparsities=}")
    row_selectors = hpart['rows']['lk'][L - 1]
    # traverse hpart tree and assign leaves set of group indices 
    # from each level to which they belong
    S = []
    for level in range(L):
        num_blocks = hpart['rows']['lk'][level].size - 1
        diff = np.diff(hpart['rows']['lk'][level])
        S += [np.repeat(np.arange(num_blocks), diff)]
    # n x (L-1)
    groups_all = np.stack(S, axis=1)
    si_groups = groups_all[row_selectors[:-1]]
    assert si_groups.shape == (row_selectors.size - 1, L) == np.unique(si_groups, axis=0).shape
    print(si_groups.shape, si_groups[-1])
    F_hpart = {"lk": hpart['rows']['lk'][:-1], "pi":hpart['rows']['pi']} 
    return row_selectors, si_groups, F_hpart


def em_algorithm(n, Y, part_sizes, F_hpart, row_selectors, si_groups, ranks, max_iter=200, 
                 eps=1e-12, printing=False):
    loglikelihoods = [-np.inf]
    rank = ranks.sum()
    F0, D0 = np.random.randn(n, rank-1), np.square(np.random.rand(n)) + 1
    for t in range(max_iter):
            Sigma0 = perm_hat_Sigma(F0, D0, F_hpart, ranks)
            lu, piv = scipy.linalg.lu_factor(Sigma0)
            obj = loglikelihood_value(Sigma0, lu, piv, Y)
            loglikelihoods += [obj]
            if printing and t % 50 == 0:
                print(f"{t=}, {obj=}")
            F1 = EM_get_F(F0, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups)
            D1 = EM_get_D(F0, F1, lu, piv, Y, ranks, part_sizes, F_hpart, row_selectors, si_groups)
            F0, D0 = F1, D1
            assert D1.min() >= -1e-8 and loglikelihoods[-2] - 1e-8 <= loglikelihoods[-1]
            if loglikelihoods[-1] - loglikelihoods[-2] < eps * abs(loglikelihoods[-2]):
                print(f"terminating at {t=}")
                break
    return loglikelihoods, F0, D0
