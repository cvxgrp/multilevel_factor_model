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


def compressed2sparse_recurrence(compressed_rec, lks, ranks):
    # convert compressed form of recurrence term to sparse
    res = []
    cumsum = 0
    for lk, rank in zip(lks, ranks):
        res += [block_diag_lk(lk, compressed_rec[:, cumsum : cumsum+rank ])]
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
        res += [ A[r1:r2]]
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
