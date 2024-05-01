import numpy as np
import mlrfit as mf
from scipy.sparse.linalg import svds, eigsh





def low_rank_approx(A, dim=None, symm=False, v0=None):
    """
    Return low rank approximation of A \approx B C^T
    """
    M =  min(A.shape[0], A.shape[1])
    if dim is None: dim = M
    dim = min(dim, min(A.shape[0], A.shape[1]))
    if dim < M:
        try:
            U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0)
        except:
            maxiter = min(A.shape) * 100
            try:
                print(f"svds fail: increase {maxiter=}")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, maxiter=maxiter)
            except:
                print(f"svds fail: decrease tol")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, tol=1e-2)
    else:
        U, sigmas, Vt = np.linalg.svd(A, full_matrices=False, hermitian=symm)
    # decreasing order of sigmas
    idx = np.argsort(sigmas)[::-1]
    sigmas = sigmas[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]
    sqrt_sigmas = np.sqrt(np.maximum(sigmas, 0))
    B = U * sqrt_sigmas
    C = Vt.T * sqrt_sigmas
    return B, C


def generate_mfmodel(true_mfm, n, F_hpart, ranks, signal_to_noise, debug=False):
    F = np.random.randn(n, ranks.sum()-1) 
    true_mfm._update_hpart(F_hpart)
    true_mfm.ranks = ranks
    true_mfm.F = F
    avg_signal_variance = np.sum(true_mfm.diag_sparse_FFt(F, F_hpart, ranks)) / n
    # expectation[D_i] = avg_signal_variance / signal_to_noise
    true_D_noise = np.random.uniform(0, (2/signal_to_noise) * avg_signal_variance, n)
    true_mfm.D = true_D_noise

    print(f"signal_var={true_mfm.diag().mean()}, noise_var={true_D_noise.mean()}")
    print(f"SNR={true_mfm.diag_sparse_FFt(F, F_hpart, ranks).mean() / true_D_noise.mean()}, {signal_to_noise=}")
    assert true_mfm.F.shape[1] == true_mfm.ranks[:-1].sum()
    if debug:
        true_sparse_F = mf.convert_compressed_to_sparse(true_mfm.F, 
                                             F_hpart, 
                                             true_mfm.ranks[:-1]).toarray()
        # permuted, ie, each group is on the block diagonal
        perm_true_covariance = true_sparse_F @ true_sparse_F.T + np.diag(true_D_noise)
        assert np.allclose(true_mfm.matrix(), perm_true_covariance[true_mfm.pi_inv, :][:, true_mfm.pi_inv])
    return true_mfm


def generate_mlr_model(n, hpart, ranks, signal_to_noise, debug=False):
    # B = [F, \sqrt{D}]
    B = np.random.randn(n, ranks.sum()) 
    true_mlr = mf.MLRMatrix(hpart=hpart, ranks=ranks, B=B, C=B)
    F_hpart = {"pi": true_mlr.hpart['rows']["pi"], 
               "lk": true_mlr.hpart['rows']["lk"][:-1]}
    true_sparse_F = mf.convert_compressed_to_sparse(true_mlr.B[:, :-1], 
                                             F_hpart, 
                                             true_mlr.ranks[:-1]).toarray()
    avg_signal_variance = np.sum(np.diag(true_sparse_F @ true_sparse_F.T)) / n
    # expectation[D_i] = avg_signal_variance / signal_to_noise
    true_D_noise = np.random.uniform(0, (2/signal_to_noise) * avg_signal_variance, n)
    true_mlr.B[:, -1] = np.sqrt(true_D_noise)
    print(f"signal_var={np.diag(true_sparse_F @ true_sparse_F.T).mean()}, noise_var={true_D_noise.mean()}")
    print(f"SNR={np.diag(true_sparse_F @ true_sparse_F.T).mean()/true_D_noise.mean()}, {signal_to_noise=}")
    if debug:
        # permuted, ie, each group is on the block diagonal
        perm_true_covariance = true_sparse_F @ true_sparse_F.T + np.diag(true_D_noise)
        assert np.allclose(true_mlr.matrix()[true_mlr.pi_rows, :][:, true_mlr.pi_cols], perm_true_covariance)
    return true_mlr, true_sparse_F, true_D_noise


def sample_data(nsamples, mfm_Sigma):
    """
    Return C: n x nsamples, where features in C are in general order
    ie, to get groups apply permutation in mfm_Sigma.pi
    """
    s = mfm_Sigma.num_factors()
    n = mfm_Sigma.F.shape[0]
    Z = np.random.randn(s, nsamples)
    E = np.sqrt(mfm_Sigma.D)[:, None] * np.random.randn(n, nsamples)
    C = mfm_Sigma.F_matvec(Z) + E
    return C[mfm_Sigma.pi_inv, :]


def generate_data(true_sparse_F, D_noise, nsamples, true_mlr):
    """
    Return C: n x nsamples, where features in C are in general order
    ie, to get groups apply permutation in true_mlr.pi_rows
    """
    n, s = true_sparse_F.shape
    Z = np.random.randn(s, nsamples)
    E = np.random.multivariate_normal(np.zeros(n), np.diag(D_noise), size=nsamples).T
    # E = np.diag(np.sqrt(D_noise)) @ np.random.randn(n, nsamples)
    # E = np.random.randn(n, nsamples) * np.sqrt(D_noise)[:, None]
    C = true_sparse_F @ Z + E
    return C[true_mlr.pi_inv_rows, :]


def print_hpart_numgroups(hpart:mf.HpartDict):
    part_sizes = []
    for level in range(len(hpart['rows']['lk'])):
        part_sizes += [hpart['rows']['lk'][level].size-1]
        print(f"{level=}, num_groups={hpart['rows']['lk'][level].size-1}, mean_size={np.diff(hpart['rows']['lk'][level]).mean():.1f}")
    return part_sizes


def valid_hpart(hpart):
    for level in range(len(hpart["rows"]["lk"])-1):
        assert np.in1d(hpart['rows']['lk'][level], hpart['rows']['lk'][level+1]).all()
    for level in range(len(hpart["cols"]["lk"])-1):
        assert np.in1d(hpart['cols']['lk'][level], hpart['cols']['lk'][level+1]).all()
        

def row_col_selections(hpart, return_groups=False):
    """
        Define row and col selectors for each row sparsity pattern of F
    """
    num_levels = len(hpart['rows']['lk'])
    L = num_levels - 1
    num_sparsities = len(hpart['rows']['lk'][L - 1]) - 1
    print(f"{num_levels=}, {num_sparsities=}")
    # row selector for each sparsity pattern
    row_selectors = hpart['rows']['lk'][L - 1]
    # traverse hpart tree and assign to each leaf  
    # set of group indices 
    # from each level to which it belongs
    S = []
    for level in range(L):
        num_blocks = hpart['rows']['lk'][level].size - 1
        diff = np.diff(hpart['rows']['lk'][level])
        # for each level assign group index to each feature
        S += [np.repeat(np.arange(num_blocks), diff)]
    # n x (L-1)
    groups_all = np.stack(S, axis=1)
    # list of groups for each sparsity pattern
    si_groups = groups_all[row_selectors[:-1]]
    assert si_groups.shape == (row_selectors.size - 1, L) == np.unique(si_groups, axis=0).shape
    print(si_groups.shape, si_groups[-1])
    F_hpart = {"lk": hpart['rows']['lk'][:-1], "pi":hpart['rows']['pi']} 
    if return_groups:
        return row_selectors, si_groups, F_hpart, groups_all
    else:
        return row_selectors, si_groups, F_hpart
    

def group_to_indices(group, part_sizes, ranks):
    """
        Given a group (corresponding to some sparsity pattern s_i)
        return the indices of nonzero columns in sparse F
    """
    cumsum = 0
    indices = []
    for level, gi in enumerate(group):
        indices += [np.arange(cumsum + gi * ranks[level], cumsum + (gi + 1) * ranks[level])]
        cumsum += ranks[level] * part_sizes[level]
    indices = np.concatenate(indices, axis=0)
    assert indices.size == ranks[:-1].sum()
    return indices
