import numpy as np
from scipy.linalg import block_diag
import scipy.sparse





def mlr_matvec(x0, B, C, hpart, ranks):
    # Compute matrix-vector with MLR matrix
    # \tilde B @ \tilde C^T @ x0
    if len(x0.shape) == 1: x0 = x0.reshape(-1, 1)
    x = x0[hpart["pi"]]
    sparse = scipy.sparse.isspmatrix(x0)
    if B.shape[0] == hpart["lk"][-1].size - 1:
        d = np.multiply(B[:, -ranks[-1]:], C[:, -ranks[-1]:]).sum(axis=1)
        L = len(hpart["lk"]) - 1
        if sparse:
            res = x.multiply(d[:, np.newaxis])
        else:
            res = d[:, np.newaxis] * x
    else:
        res = np.zeros(x.shape)
        L = len(hpart["lk"])
 
    for level in range(L):
        lk = hpart["lk"][level]
        num_blocks = lk.size - 1 
        for block in range(num_blocks):
            r1, r2 = lk[block], lk[block+1]
            res[r1:r2] += B[r1:r2, ranks[:level].sum():ranks[:level+1].sum()] @ \
                                (C[r1:r2, ranks[:level].sum():ranks[:level+1].sum()].T @ x[r1:r2])
    return res[hpart["pi_inv"]]


def mult_blockdiag_refined_CtB(C, lk_C, B, lk_B):
    """
    multiply blockdiagonal matrix by blockdiagonal matrix
    blockdiagonal(Ct) @ blockdiagonal(B)
    Ct's column sparity is refined by B's row sparsity
    C, B given in compressed format
    return ( r_C x num_blocks_B * r_B)
    """
    assert lk_C.size <= lk_B.size and lk_B[-1] in B.shape and lk_C[-1] in C.shape
    num_blocks_B = lk_B.size - 1
    res = np.zeros((C.shape[1], B.shape[1] * (lk_B.size - 1)))
    # make C block diagonal by splitting rows
    for block_B in range(num_blocks_B):
        r1, r2 = lk_B[block_B], lk_B[block_B+1]
        res[:, block_B * B.shape[1] : (block_B+1) * B.shape[1]] = C[r1:r2].T @ B[r1:r2]
    return res


def mult_blockdiag_refined_BCt(B, lk_B, C, lk_C):
    """
    multiply blockdiagonal matrix by blockdiagonal matrix
    blockdiagonal(B) @ blockdiagonal(Ct)
    B's column sparity is refined by Ct's row sparsity
    B, C given in compressed format
    """
    assert lk_B.size <= lk_C.size and lk_B[-1] in B.shape and lk_C[-1] in C.shape, \
        print(lk_B.size, lk_C.size, lk_B[-1], B.shape, lk_C[-1], C.shape)
    num_blocks_C = lk_C.size - 1
    res = np.zeros((B.shape[0], C.shape[0]))
    # make B block diagonal by splitting columns
    for block_C in range(num_blocks_C):
        r1, r2 = lk_C[block_C], lk_C[block_C+1]
        res[:, r1 : r2] = B[:, block_C * C.shape[1] : (block_C+1) * C.shape[1]] @ C[r1:r2].T
    return res.T


def block_diag_lk_t(lk:np.array, A:np.ndarray):
    # return blockdiagonal(A)
    res = []
    num_blocks = lk.size - 1 
    for block in range(num_blocks):
        r1, r2 = lk[block], lk[block+1]
        res += [A[:, r1:r2]]
    return block_diag(*res)
    

def mlr_level_matvec_base(l1, B_l1, C_l1, l2, B_l2, C_l2, hpart):
    # Compute A_l1 @ \tilde A_l2
    assert l1 <= l2
    B = B_l1 + 0
    if hpart["lk"][l2].size-1 == C_l1.shape[0]:
        # \tilde A_l2 is diagonal
        d = np.multiply(B_l2, C_l2).sum(axis=1).reshape(-1, 1)
        C = d * C_l1
    else:
        #  (r_C_l1 x num_blocks_l2 * r_B_l2)
        C_l1t_B_l2 = mult_blockdiag_refined_CtB(C_l1, hpart['lk'][l1], 
                                                B_l2, hpart['lk'][l2])
        indices_Ct2B = np.searchsorted(hpart['lk'][l2], hpart['lk'][l1], side='left') * B_l2.shape[1]
        C = mult_blockdiag_refined_BCt(C_l1t_B_l2, indices_Ct2B, 
                                    C_l2, hpart['lk'][l2])
    return B, C


def mlr_level_matvec(l1, B_l1, C_l1, l2, B_l2, C_l2, hpart):
    # Compute A_l1 @ \tilde A_l2^T 
    if l1 <= l2:
        return mlr_level_matvec_base(l1, B_l1, C_l1, l2, B_l2, C_l2, hpart)
    else:
        C, B =  mlr_level_matvec_base(l2, C_l2, B_l2, l1, C_l1, B_l1, hpart)
        return B, C
    

def concatenate_factors_all_levels(B1, ranks1, B2, ranks2):
    # concatenate factors in B1, B2 
    # reorganizing factors contiguously for each level
    B = np.zeros((B1.shape[0], ranks1.sum()+ranks2.sum()))
    ranks = ranks1 + ranks2
    for l in range(ranks1.size):
        r1, r2 = ranks1[l], ranks2[l]
        B[:, ranks[:l].sum():ranks[:l].sum()+r1] = B1[:, ranks1[:l].sum():ranks1[:l+1].sum()]
        B[:, ranks[:l].sum()+r1 :ranks[:l].sum()+r1+r2] = B2[:, ranks2[:l].sum():ranks2[:l+1].sum()]
    return B, ranks


def mlr_mlr_sum(B1, C1, ranks1, B2, C2, ranks2, hpart):
    B, ranks = concatenate_factors_all_levels(B1, ranks1, B2, ranks2)
    C, ranks = concatenate_factors_all_levels(C1, ranks1, C2, ranks2)
    # combine factors on the last level if it is diagonal 
    if B.shape[0] == hpart["lk"][-1].size - 1 and ranks[-1] >= 2:
        d = np.multiply(B[:, -ranks[-1]:], C[:, -ranks[-1]:]).sum(axis=1)
        ranks[-1] = 1
        B = B[:, :ranks.sum()]
        B[:, -1] = 1
        C = C[:, :ranks.sum()]
        C[:, -1] = d
    assert B.shape[1] == C.shape[1] == ranks.sum()
    return B, C, ranks


def mlr_mlr_matmul(B1, C1, ranks1, B2, C2, ranks2, hpart):
    """
    Product of two MLR matrices with the same hpart
    is an MLR with MLR-rank that is a sum of MLR-ranks of two matrices
    """
    ranks = ranks1 + ranks2
    if B1.shape[0] == hpart["lk"][-1].size - 1:
        ranks[-1] = ranks1[-1]
    L = ranks.size
    n = B1.shape[0]
    B, C = np.zeros((n, ranks.sum())), np.zeros((n, ranks.sum()))

    for l in range(L-1): 
        C_l1 = C1[:,ranks1[:l].sum():ranks1[:l+1].sum()]
        B_l2 = B2[:,ranks2[:l].sum():ranks2[:l+1].sum()]
        
        r1 = ranks1[l]
        r2 = ranks2[l]
        B[:, ranks[:l].sum():ranks[:l].sum()+r1] = B1[:, ranks1[:l].sum():ranks1[:l+1].sum()]
        C[:, ranks[:l].sum()+r1 :ranks[:l].sum()+r1+r2] = C2[:, ranks2[:l].sum():ranks2[:l+1].sum()]
        
        for l_p in range(l, L):
            B_l_p2, C_l_p2 = B2[:,ranks2[:l_p].sum():ranks2[:l_p+1].sum()], C2[:,ranks2[:l_p].sum():ranks2[:l_p+1].sum()]
            _, C_l_p = mlr_level_matvec(l, 0, C_l1, l_p, B_l_p2, C_l_p2, hpart)
            C[:, ranks[:l].sum() :ranks[:l].sum()+r1] += C_l_p
            if l_p >= l+1:
                B_l_p1, C_l_p1 = B1[:,ranks1[:l_p].sum():ranks1[:l_p+1].sum()], C1[:,ranks1[:l_p].sum():ranks1[:l_p+1].sum()]
                B_l_p, _ = mlr_level_matvec(l_p, B_l_p1, C_l_p1, l, B_l2, 0, hpart)
                B[:, ranks[:l].sum()+r1 : ranks[:l].sum()+r1+r2] += B_l_p


    B_L, C_L = mlr_level_matvec(L-1, B1[:, -ranks1[L-1]:], C1[:, -ranks1[L-1]:], 
                                L-1, B2[:, -ranks2[L-1]:], C2[:, -ranks2[L-1]:], hpart)

    B[:, -ranks[-1]:] += B_L
    C[:, -ranks[-1]:] += C_L
    return B, C, ranks
