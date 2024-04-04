import numpy as np
import mlrfit as mf






def generate_mlr_model(n, hpart, ranks, signal_to_noise):
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
    # permuted, ie, each group is on the block diagonal
    perm_true_covariance = true_sparse_F @ true_sparse_F.T + np.diag(true_D_noise)
    assert np.allclose(true_mlr.matrix()[true_mlr.pi_rows, :][:, true_mlr.pi_cols], perm_true_covariance)
    return true_mlr, true_sparse_F, true_D_noise


def generate_data(true_sparse_F, D_noise, nsamples, true_mlr):
    """
    Return C: n x nsamples
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
