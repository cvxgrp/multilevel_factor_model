import numpy as np

import mlrfit as mf
import numba as nb

from scipy.sparse import coo_matrix
from scipy.linalg import block_diag, pinvh





def mult_blockdiag_refined_AtB(A, lk_A, B, lk_B):
    # multiply blockdiagonal matrix by blockdiagonal matrix
    # blockdiagonal(At) @ blockdiagonal(B)
    # At has column sparity that is a refinement of B row sparsity
    # A, B given in compressed format
    assert lk_A.size >= lk_B.size and lk_B[-1] == B.shape[0] and lk_A[-1] in A.shape
    num_blocks_A = lk_A.size - 1
    res = np.zeros((A.shape[1] * (lk_A.size - 1), B.shape[1]))
    # decide whether make A block diagonal by splitting rows or columns
    for block_A in range(num_blocks_A):
        r1, r2 = lk_A[block_A], lk_A[block_A+1]
        res[block_A * A.shape[1] : (block_A+1) * A.shape[1]] = A[r1:r2].T @ B[r1:r2]
    return res


def mult_blockdiag_refined_AB(A, lk_A, B, lk_B):
    # multiply blockdiagonal matrix by blockdiagonal matrix
    # blockdiagonal(A) @ blockdiagonal(B)
    # A has column sparity that is a refinement of B row sparsity
    # A, B given in compressed format
    assert lk_A.size >= lk_B.size and lk_B[-1] == B.shape[0] and lk_A[-1] in A.shape
    num_blocks_A = lk_A.size - 1
    res = np.zeros((A.shape[0], B.shape[1]))
    # decide whether make A block diagonal by splitting rows or columns
    for block_A in range(num_blocks_A):
        r1, r2 = lk_A[block_A], lk_A[block_A+1]
        res[r1 : r2] = A[r1:r2] @ B[block_A * A.shape[1] : (block_A+1) * A.shape[1]]
    return res


def fast_SMW_inverse(ranks, v0, F_Lm1, D, F_hpart):
    sqrt_D = np.power(D, -0.5)[:, np.newaxis]
    res = fast_SMW_inverse_basic(ranks, sqrt_D[F_hpart['pi_inv']] * v0, 
                                sqrt_D * F_Lm1, np.ones(D.size), F_hpart)
    return sqrt_D[F_hpart['pi_inv']] * res


def fast_SMW_inverse_basic(ranks, v0, F_Lm1, D, F_hpart):
    pi = F_hpart['pi']
    if "pi_inv" not in F_hpart:
        pi_inv, _ = mf.inv_permutation(pi, pi)
    else:
        pi_inv = F_hpart['pi_inv']
    v = v0.flatten()[pi]
    v_F_Lm1 = np.concatenate([v[:, np.newaxis], F_Lm1], axis=1)
    prev_l_recurrence = (1/D[:, np.newaxis]) * v_F_Lm1
    n = v.size
    L = len(F_hpart['lk']) + 1
    for level in reversed(range(1, L-1)):
        pl = F_hpart['lk'][level].size - 1
        rl = ranks[level]
        # M0 same sparsity as Fl
        M0 = prev_l_recurrence[:, -ranks[level]:]
        # M1 = M0.T @ rec_term, same sparsity as rec_term
        M1 = np.zeros((rl * pl, 1 + ranks[:level].sum()))
        for lp in range(level):
            if lp == 0:
                M1[:, :1] = mult_blockdiag_refined_AtB(M0, 
                                                        F_hpart['lk'][level], 
                                                        v.reshape(-1, 1), 
                                                        F_hpart['lk'][lp])
            M1[:, 1+ranks[:lp].sum() : 1+ranks[:lp+1].sum()] = mult_blockdiag_refined_AtB(M0, 
                                                                                F_hpart['lk'][level], 
                                                                                F_Lm1[:, ranks[:lp].sum():ranks[:lp+1].sum()], 
                                                                                F_hpart['lk'][lp])
        M1_lks = [np.searchsorted(F_hpart['lk'][level], lk_B, side='left') * rl for lk_B in F_hpart['lk'][:level]]
        # M2 = (I + Fl^T M0)^{-1}, blockdiagonal with pl blocks of size (rl x rl)
        FlTM0 = mult_blockdiag_refined_AtB(F_Lm1[:, ranks[:level].sum() : ranks[:level+1].sum()], 
                                        F_hpart['lk'][level], 
                                        M0, 
                                        F_hpart['lk'][level])
        M2 = np.zeros((pl*rl, rl))
        for k in range(pl):
            np.fill_diagonal(FlTM0[k*rl : (k+1)*rl], FlTM0[k*rl : (k+1)*rl].diagonal() + 1)
            M2[k*rl : (k+1)*rl] = pinvh(FlTM0[k*rl : (k+1)*rl])
        del FlTM0
        # M3 = M2 @ M1, same sparsity as M1
        M3 = np.zeros((rl * pl, 1 + ranks[:level].sum()))
        for lp in range(level):
            if lp == 0:
                M3[:, :1] = mult_blockdiag_refined_AtB(M2, 
                                                        np.linspace(0, pl*rl, num=pl+1, endpoint=True, dtype=int), 
                                                        M1[:, :1], 
                                                        M1_lks[lp])
            M3[:, 1+ranks[:lp].sum():1+ranks[:lp+1].sum()] = mult_blockdiag_refined_AtB(M2, 
                                                                                        np.linspace(0, pl*rl, num=pl+1, endpoint=True, dtype=int), 
                                                                                        M1[:,1+ranks[:lp].sum():1+ranks[:lp+1].sum()], 
                                                                                        M1_lks[lp])
        del M1, M2
        # M4 = M0 @ M3, same sparsity as current rec_term
        M4 = np.zeros((n, 1 + ranks[:level].sum()))
        for lp in range(level):
            if lp == 0:
                M4[:, :1] = mult_blockdiag_refined_AB(M0, 
                                                        F_hpart["lk"][level], 
                                                        M3[:, :1], 
                                                        M1_lks[lp])
            M4[:, 1+ranks[:lp].sum() : 1+ranks[:lp+1].sum()] = mult_blockdiag_refined_AB(M0, 
                                                                                        F_hpart["lk"][level], 
                                                                                        M3[:,1+ranks[:lp].sum():1+ranks[:lp+1].sum()], 
                                                                                        M1_lks[lp])
        del M0, M3
        # M5 
        prev_l_recurrence = prev_l_recurrence[:, :1+ranks[:level].sum()] - M4
        del M4
    # final
    level = 0
    pl = F_hpart['lk'][level].size - 1
    rl = ranks[level]
    # M0 same sparsity as Fl
    M0 = prev_l_recurrence[:, -ranks[level]:]
    # M1 = M0.T @ rec_term, same sparsity as rec_term
    M1 = mult_blockdiag_refined_AtB(M0, 
                                    F_hpart['lk'][level], 
                                    v[:, np.newaxis], 
                                    F_hpart['lk'][0])
    # M2 = (I + Fl^T M0)^{-1}, blockdiagonal with pl blocks of size (rl x rl)
    FlTM0 = mult_blockdiag_refined_AtB(F_Lm1[:, :ranks[:level+1].sum()], 
                                        F_hpart['lk'][level], 
                                        M0, 
                                        F_hpart['lk'][level])
    M2 = pinvh(np.eye(rl) + FlTM0)
    del FlTM0
    # M3 = M2 @ M1, same sparsity as M1
    M3 = M2 @ M1
    del M2, M1
    # M4 = M0 @ M3, same sparsity as current rec_term
    M4 = M0 @ M3
    del M0, M3
    # M5 
    prev_l_recurrence = prev_l_recurrence[:, :1] - M4 
    del M4
    return prev_l_recurrence[pi_inv]


def mfm_matvec(F_compressed, D, F_hpart, ranks, x0):
    # return \Sigma x 
    pi, pi_inv = F_hpart['pi'], F_hpart['pi_inv']
    if len(x0.shape) == 1: x0 = x0.reshape(-1, 1)
    x = x0[pi]
    res = D[:, np.newaxis] * x
    for level in range(len(F_hpart["lk"])):
        lk = F_hpart["lk"][level]
        num_blocks = lk.size - 1 
        for block in range(num_blocks):
            r1, r2 = lk[block], lk[block+1]
            res[r1:r2] += F_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()] @ \
                (F_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()].T @ x[r1:r2])
    return res[pi_inv]


def iterative_refinement(ranks, v, F_Lm1, D, F_hpart, eps=1e-11, max_iter=20, printing=False):
    v_norm = np.linalg.norm(v)
    if len(v.shape) == 1: v = v.reshape(-1, 1)
    residual = v_norm + 0
    t = 0
    x = np.zeros((v.size, 1))
    Ax = x + 0
    while residual / v_norm > eps and t < max_iter:
        delta = fast_SMW_inverse(ranks, v - Ax, F_Lm1, D, F_hpart)
        x += delta
        Ax = mfm_matvec(F_Lm1, D, F_hpart, ranks, x)
        residual = np.linalg.norm(Ax - v)
        t += 1
        if printing:
            print(residual/v_norm)
    if printing and residual / v_norm > eps:
        print(f"terminated with {residual/v_norm=}")
    return x


def inv_rec_term_to_sparse(compressed_rec, lks, ranks):
    # convert compressed form of recurrence term in the SMW inverse computation to sparse
    res = [compressed_rec[:, :1]]
    cumsum = 0
    for lk, rank in zip(lks, ranks):
        res += [block_diag_lk(lk, compressed_rec[:, 1 + cumsum : 1 + cumsum+rank ])]
        cumsum += rank 
    return np.concatenate(res, axis=1)


def block_diag_AB(lk:np.array, A:np.ndarray, B:np.ndarray):
    # return blockdiagonal(A) @ blockdiagonal(B)
    res = []
    num_blocks = lk.size - 1
    for block in range(num_blocks):
        r1, r2 = lk[block], lk[block+1]
        res += [ A[r1:r2] @ B[:, r1:r2] ]
    return block_diag(*res)


def block_diag_lk(lk:np.array, A:np.ndarray):
    # return blockdiagonal(A)
    res = []
    num_blocks = lk.size - 1 
    for block in range(num_blocks):
        r1, r2 = lk[block], lk[block+1]
        res += [A[r1:r2]]
    return block_diag(*res)


@nb.njit(parallel=True)
def jit_mult_blockdiag_refined_AtB(A, lk_A, B, lk_B):
    # multiply blockdiagonal matrix by blockdiagonal matrix
    # blockdiagonal(At) @ blockdiagonal(B)
    # At has column sparity that is a refinement of B row sparsity
    # A, B given in compressed format
    num_blocks_A = lk_A.size - 1
    res = np.zeros((A.shape[1] * (lk_A.size - 1), B.shape[1]))
    # decide whether make A block diagonal by splitting rows or columns
    for block_A in nb.prange(num_blocks_A):
        r1, r2 = lk_A[block_A], lk_A[block_A+1]
        res[block_A * A.shape[1] : (block_A+1) * A.shape[1]] = A[r1:r2].T @ B[r1:r2]
    return res

