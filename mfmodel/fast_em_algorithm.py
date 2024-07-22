from scipy.linalg import block_diag
import scipy.sparse
import mlrfit as mf
import numpy as np


from mfmodel.utils import *
from mfmodel.inverse import *
from mfmodel.mfmodel import *



def fit(Y, ranks, hpart, init_type="Y", max_iter=100, eps=1e-8, printing=False, freq=10):
    fitted_mfm = MFModel(hpart=hpart, ranks=ranks)
    fitted_mfm.init_FD(ranks, hpart, init_type=init_type, Y=Y)
    F0, D0 = fitted_mfm.F, fitted_mfm.D
    n = F0.shape[0]
    permuted_F_hpart = {"pi_inv":np.arange(n), "pi":np.arange(n), "lk":hpart["lk"]}
    row_selectors, si_groups, _ = si_row_col(hpart)
    F0, D0, loglikelihoods = fast_em_algorithm(Y,  F0, D0, permuted_F_hpart, row_selectors, si_groups, 
                                               ranks, lls=True, max_iter=max_iter, eps=eps, printing=printing, freq=freq) 
    fitted_mfm.F, fitted_mfm.D = F0, D0 
    fitted_mfm.inv_coefficients(det=True) 
    return fitted_mfm, loglikelihoods


def fast_EM_intermediate_matrices(F_Lm1, D, F_hpart, Y, ranks, si, si_groups, row_selectors, mfm_Sigma=None):
    tilde_F_ci = get_sparse_F_si_col_sparsity(F_Lm1, ranks, F_hpart, si_groups[si]) # n x (r-1)
    if mfm_Sigma is None:
        mfm_Sigma = MFModel(hpart=F_hpart, ranks=ranks, F=F_Lm1, D=D)
        mfm_Sigma.inv_coefficients() 
    return fast_EM_intermediate_matrices_base(mfm_Sigma, tilde_F_ci, Y, ranks, si, row_selectors)
    

def fast_EM_intermediate_matrices_base(mfm_Sigma, tilde_F_ci, Y, ranks, si, row_selectors):
    r1, r2 = row_selectors[si: si+2]
    N, n = Y.shape
    Sigma0_inv_F_ci = np.zeros(tilde_F_ci.shape) 
    Sigma0_inv_F_ci = mfm_Sigma.solve(tilde_F_ci)
    F_ciT_Sigma0_inv_F_ci = tilde_F_ci.T @ Sigma0_inv_F_ci # (r-1) x (r-1)
    del tilde_F_ci
    Y_Sigma0_inv_F_ci = Y @ Sigma0_inv_F_ci # N x (r-1)
    ci_W_ci = N * (np.eye(ranks[:-1].sum()) - F_ciT_Sigma0_inv_F_ci) + Y_Sigma0_inv_F_ci.T @ Y_Sigma0_inv_F_ci
    del F_ciT_Sigma0_inv_F_ci
    ri_Vt_ci_t = Y_Sigma0_inv_F_ci.T @ Y[:, r1:r2] # (r-1) x (r2-r1)
    return ri_Vt_ci_t, ci_W_ci, r1, r2


def fast_EM_get_F(F0, D0, Y, ranks, F_hpart, row_selectors, si_groups, mfm_Sigma=None, return_mfm=False):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    F1 = np.zeros(F0.shape)
    num_sparsities = row_selectors.size - 1
    if mfm_Sigma is None:
        mfm_Sigma = MFModel(hpart=F_hpart, ranks=ranks, F=F0, D=D0)
        mfm_Sigma.inv_coefficients()
    for si in range(num_sparsities):
        tilde_F_ci = get_sparse_F_si_col_sparsity(F0, ranks, F_hpart, si_groups[si]) # n x (r-1)
        ri_Vt_ci_t, ci_W_ci, r1, r2 = fast_EM_intermediate_matrices_base(mfm_Sigma, tilde_F_ci, Y, ranks, si, row_selectors)
        F1[r1:r2, :] = np.linalg.solve(ci_W_ci, ri_Vt_ci_t).T
    if return_mfm:
        return F1, mfm_Sigma
    else:
        return F1


def fast_EM_get_D(F0, D0, F1, Y, ranks, F_hpart, row_selectors, si_groups, mfm_Sigma=None, return_mfm=False):
    """
        Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
    """
    N, n = Y.shape
    num_sparsities = row_selectors.size - 1
    D1 = np.zeros(n)
    if mfm_Sigma is None:
        mfm_Sigma = MFModel(hpart=F_hpart, ranks=ranks, F=F0, D=D0)
        mfm_Sigma.inv_coefficients() 
    sparse = scipy.sparse.isspmatrix(Y)
    for si in range(num_sparsities):
        tilde_F_ci = get_sparse_F_si_col_sparsity(F0, ranks, F_hpart, si_groups[si]) # n x (r-1)
        ri_Vt_ci_t, ci_W_ci, r1, r2 = fast_EM_intermediate_matrices_base(mfm_Sigma, tilde_F_ci, Y, ranks, si, row_selectors)
        ri_F1_ciT = F1[r1:r2]
        if sparse:
            ri_YtY_ri = scipy.sparse.csr_array.sum(scipy.sparse.csr_array.multiply(Y[:, r1:r2], Y[:, r1:r2]), axis=0)
        else:
            ri_YtY_ri = np.einsum('ij,ji->i', Y[:, r1:r2].T, Y[:, r1:r2])
        D1[r1:r2] = (1/N) * ( ri_YtY_ri \
                            - 2 * np.einsum('ij,ji->i', ri_F1_ciT, ri_Vt_ci_t) \
                            + np.einsum('ij,jk,ki->i', ri_F1_ciT, ci_W_ci, ri_F1_ciT.T))
    if return_mfm:
        return D1, mfm_Sigma
    else:
        return D1


def fast_em_algorithm(Y, F0, D0, F_hpart, row_selectors, si_groups, ranks, lls=False, max_iter=200, eps=1e-12, 
                      freq=50, printing=False):
        """
            Fast EM algorithm
            Y: N x n has already permuted columns, ie, features ordered wrt F_hpart
        """
        loglikelihoods = [-np.inf]
        obj = loglikelihoods[-1]
        N = Y.shape[0]
        for t in range(max_iter):
            mfm_Sigma0 = MFModel(hpart=F_hpart, ranks=ranks, F=F0, D=D0)
            mfm_Sigma0.inv_coefficients(det=True)
            if lls:
                obj = fast_loglikelihood_value(mfm_Sigma0, Y[:, mfm_Sigma0.pi_inv])
                loglikelihoods += [obj]
                
            if printing and t % freq == 0:
                print(f"{t=}, {obj=}")

            F1 = fast_EM_get_F(F0, D0, Y, ranks, F_hpart, row_selectors, si_groups, mfm_Sigma=mfm_Sigma0)
            D1 = fast_EM_get_D(F0, D0, F1, Y, ranks, F_hpart, row_selectors, si_groups, mfm_Sigma=mfm_Sigma0)
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


def full_mlr_frob_norm(B_L, C_L, ranks, hpart, si_groups_L):
    """
    Return \|B C^T \|_F
    """
    res = 0
    n = B_L.shape[0]
    for si in range(n):
        r1, r2 = si, si+1 # contiguous range for group si
        tilde_B_ci = get_sparse_F_si_col_sparsity(B_L, ranks, hpart, si_groups_L[si]) # n x (r-1)
        assert tilde_B_ci.shape == (n, ranks.sum()), print(tilde_B_ci.shape, (n, ranks.sum()))
        ci_Bt_B_ci = tilde_B_ci.T @ tilde_B_ci
        del tilde_B_ci
        ri_C_ci = get_sparse_F_si_col_sparsity(C_L, ranks, hpart, si_groups_L[si])[r1:r2] # (r2-r1) x (r-1)
        res += np.einsum('ik,ik->i', (ri_C_ci @ ci_Bt_B_ci), ri_C_ci).sum()
        del ri_C_ci
    return np.sqrt(max(0, res))


def full_mlr_trace(B_L, C_L, ranks, hpart, row_selectors, si_groups):
    """
    Return trace(B C^T)
    """
    num_sparsities = row_selectors.size - 1
    res = (B_L[:,-1] * C_L[:, -1]).sum()
    for si in range(num_sparsities):
        r1, r2 = row_selectors[si: si+2] # contiguous range for group si
        ri_C_ci = get_sparse_F_si_col_sparsity(C_L, ranks, hpart, si_groups[si])[r1:r2] # (r2-r1) x (r-1)
        ri_B_ci = get_sparse_F_si_col_sparsity(B_L, ranks, hpart, si_groups[si])[r1:r2] # (r2-r1) x (r-1)
        res += np.einsum('ik,ik->i', ri_B_ci, ri_C_ci).sum()
        del ri_B_ci, ri_C_ci
    return res


def fast_exp_loglikelihood_value(B_true, fitted_mfm, ranks, hpart, row_selectors, si_groups):
    """
        Expected log-likelihood under the true model
    """
    n = B_true.shape[0]
    logdet = fitted_mfm.logdet  
    # \tilde \Sigma^{-1} \Sigma
    B_prod, C_prod, ranks_prod = mlr_mlr_matmul(fitted_mfm.inv_B, fitted_mfm.inv_C, fitted_mfm.inv_ranks, 
                                                B_true, B_true, ranks, fitted_mfm.inv_hpart) 
    tr_fitted_Sigma_inv_true_Sigma = full_mlr_trace(B_prod, C_prod, ranks_prod, hpart, 
                                                    row_selectors, si_groups)
    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * logdet - (1/2) * tr_fitted_Sigma_inv_true_Sigma 
    return obj


def fast_exp_true_loglikelihood_value(true_mfm):
    """
        True expected log-likelihood 
    """
    n = true_mfm.F.shape[0]
    logdet = true_mfm.logdet 
    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * logdet - (1/2) * n
    return obj


def fast_loglikelihood_value(mfm_Sigma0, Y):
    """
        Average log-likelihood of observed data
    """
    N, n = Y.shape
    Sigma_inv_Yt = np.zeros((n, N))
    logdet = mfm_Sigma0.logdet
    if scipy.sparse.isspmatrix(Y):
        Sigma_inv_Yt = mfm_Sigma0.solve(Y.transpose().tocsr())
        print(Sigma_inv_Yt.shape, Y.shape)
        tr_Sigma_inv_YtY = scipy.sparse.coo_array.sum(Sigma_inv_Yt.multiply(Y))
    else:
        Sigma_inv_Yt = mfm_Sigma0.solve(Y.T)
        tr_Sigma_inv_YtY = np.einsum('ij,ji->i', Y, Sigma_inv_Yt).sum()
    obj = - (n/2) * np.log(2 * np.pi) - (1/2) * (logdet) - (0.5 / N) * tr_Sigma_inv_YtY
    return obj
    

def fast_loglikelihood_value2(F0, D0, Y, ranks, F_hpart, return_mfm=False, mfm_Sigma0=None):
    """
        Average log-likelihood of observed data
    """
    if mfm_Sigma0 is None:
        mfm_Sigma0 = MFModel(hpart=F_hpart, ranks=ranks, F=F0, D=D0)
        mfm_Sigma0.inv_coefficients(det=True) 

    obj = fast_loglikelihood_value(mfm_Sigma0, Y)

    if return_mfm:
        return obj, mfm_Sigma0
    else:
        return obj


def fast_frob_fit_loglikehood(undermuted_A, Y, F_hpart, hpart, ranks, printing=True, eps_ff=1e-3):
    hat_A = mf.MLRMatrix()
    hat_A.hpart = hpart
    losses = hat_A.factor_fit(undermuted_A, ranks, hat_A.hpart, eps_ff=eps_ff, PSD=True, freq=1, \
                                    printing=printing, max_iters_ff=50, symm=True)
    F_frob, D_frob = hat_A.B[:, :-1], np.square(hat_A.B[:, -1])
    frob_mfm = MFModel(hpart=F_hpart, ranks=ranks, F=F_frob, D=D_frob)
    frob_mfm.D = np.maximum(1e-7, frob_mfm.D)
    frob_mfm.inv_coefficients(det=True)
    return frob_mfm, losses
