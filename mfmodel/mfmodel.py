import copy
from typing import List, Tuple, Callable, TypedDict, List, Set, Optional 

import numpy as np
from scipy.linalg import block_diag

import mlrfit as mf


from mfmodel.utils import *
from mfmodel.inverse import *
from mfmodel.mlr_matmul import *
# from mfmodel.fast_em_algorithm import *




"""
Multilevel Factor Model class
"""
class MFModel:
    def __init__(self, hpart:Optional[mf.EntryHpartDict ]=None, ranks:Optional[np.ndarray]=None, \
                       F:Optional[np.ndarray]=None, D:Optional[np.ndarray]=None,
                       debug=False):
        """
        Use provided F and D as a warm start
        hpartition: dict
            {'pi':np.ndarray(m), 'lk':List[List[int]]}
             hierarchical partitioning containing block segments for every level
        F: np.ndarray(n, r) 
            compressed format of F_lk
        D: np.ndarray(n) 
            vector format of diagonal matrix D
        """
        self.F = F
        self.D = D
        self.inv_B, self.inv_C, self.si_groups, self.row_selectors = None, None, None, None
        self.ranks = ranks
        self.debug = debug
        if hpart is not None:
            self._update_hpart(hpart)


    def _update_hpart(self, hpart:mf.EntryHpartDict):
        self.hpart = hpart
        self.pi = hpart['pi']
        self.pi_inv = mf.inv_permutation(self.pi, self.pi)[0]


    def init_FD(self, ranks:np.ndarray, hpart:mf.EntryHpartDict, init_type='Y', Y=None):
        """
        Initialize B and C given ranks and hpartition
        """
        n = hpart['pi'].size
        r = ranks.sum()
        if init_type == 'random':
            F = np.random.randn(n, r-1)
            D = np.random.rand(n) + 1e-3
        elif init_type == 'Y':
            N = Y.shape[0] # Y is already permuted to put factors on blockdiagonal
            _, C = low_rank_approx(Y, dim=ranks[:-1].sum(), symm=False)
            F = C / np.sqrt(N)
            D = np.maximum(np.einsum('ij,ji->i', Y.T, Y) / N - self.diag_sparse_FFt(F, hpart, ranks), 1e-4)
        self.F, self.D = F, D
        return F, D
    

    def diag(self):
        # return diagonal of Sigma
        return (self.diag_sparse_FFt(self.F, self.hpart, self.ranks) + self.D)[self.pi_inv]
    

    def diag_inv(self):
        # return diagonal of Sigma^{-1}
        return self.diag_sparse_BCt(self.inv_B, self.inv_C, self.inv_hpart, self.inv_ranks)[self.pi_inv]
    

    def diag_sparse_BCt(self, B_compressed, C_compressed, hpart, ranks):
        # F_compressed: n x (r-1) 
        # return diag(\tilde F \tilde F^T) without permutation
        res = np.zeros(hpart['pi'].size)
        for level in range(len(hpart["lk"])):
            lk = hpart["lk"][level]
            num_blocks = lk.size - 1 
            for block in range(num_blocks):
                r1, r2 = lk[block], lk[block+1]
                res[r1:r2] += np.einsum('ij,ji->i', B_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()],
                                                    C_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()].T)
        return res


    def diag_sparse_FFt(self, F_compressed, hpart, ranks):
        # F_compressed: n x (r-1) 
        # return diag(\tilde F \tilde F^T) without permutation
        res = np.zeros(hpart['pi'].size)
        for level in range(len(hpart["lk"])):
            lk = hpart["lk"][level]
            num_blocks = lk.size - 1 
            for block in range(num_blocks):
                r1, r2 = lk[block], lk[block+1]
                res[r1:r2] += np.einsum('ij,ji->i', F_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()],
                                                    F_compressed[r1:r2, ranks[:level].sum():ranks[:level+1].sum()].T)
        return res


    def matvec(self, x0):
        # Compute \Sigma @ x0
        if len(x0.shape) == 1: x0 = x0.reshape(-1, 1)
        x = x0[self.pi]
     
        res = self.D[:, np.newaxis] * x
        for level in range(len(self.hpart["lk"])):
            lk = self.hpart["lk"][level]
            num_blocks = lk.size - 1 
            for block in range(num_blocks):
                r1, r2 = lk[block], lk[block+1]
                res[r1:r2] += self.F[r1:r2, self.ranks[:level].sum():self.ranks[:level+1].sum()] @ \
                                    (self.F[r1:r2, self.ranks[:level].sum():self.ranks[:level+1].sum()].T @ x[r1:r2])

        return res[self.pi_inv]
    

    def F_matvec(self, z):
        # Compute F @ z, permuted
        if len(z.shape) == 1: z = z.reshape(-1, 1)
        res = np.zeros((self.F.shape[0], z.shape[1]))
        count = 0
        for level in range(len(self.hpart["lk"])):
            lk = self.hpart["lk"][level]
            num_blocks = lk.size - 1 
            for block in range(num_blocks):
                r1, r2 = lk[block], lk[block+1]
                c1, c2 = self.ranks[:level].sum(), self.ranks[:level+1].sum()
                res[r1:r2] += self.F[r1:r2, c1:c2] @ z[count: count + self.ranks[level]]
                count += self.ranks[level]
        # assert count == self.num_factors()
        return res
    
    
    def num_factors(self):
        # return number of unique factors
        return (np.array([self.hpart["lk"][l].size-1 for l in range(len(self.hpart["lk"]))]) * self.ranks[:-1]).sum()
    
    
    def matrix(self):
        # return \Sigma matrix
        perm_hat_A = self._compute_perm_symm_hat_A(self.F, self.D, self.hpart, self.ranks)
        # pi_inv to permute \hat \Sigma_l from block diagonal in order approximating \Sigma
        hat_A = perm_hat_A[self.pi_inv, :][:, self.pi_inv]
        return hat_A
    

    def inv(self):
        # return \Sigma^{-1} matrix
        perm_hat_A = self._compute_perm_hat_A(self.inv_B, self.inv_C, self.inv_hpart, self.inv_ranks)
        # pi_inv to permute \hat A_l from block diagonal in order approximating \Sigma^{-1}
        hat_A = perm_hat_A[self.pi_inv, :][:, self.pi_inv]
        return hat_A
    

    def _compute_perm_hat_A(self, B:np.ndarray, C:np.ndarray, hpart:mf.EntryHpartDict , ranks:np.ndarray):
        """
        Compute permuted hat_A with each Sigma_level being block diagonal matrix 
        """
        num_levels = ranks.size
        perm_hat_A = np.zeros((B.shape[0], C.shape[0]))
        for level in range(num_levels):
            perm_hat_A += self._block_diag_BCt(level, hpart, B[:,ranks[:level].sum():ranks[:level+1].sum()],
                                                             C[:,ranks[:level].sum():ranks[:level+1].sum()])
        return perm_hat_A
    

    def _block_diag_BCt(self, level:int, hpart:mf.EntryHpartDict, B_level:np.ndarray, C_level:np.ndarray):
        A_level = []
        num_blocks = len(hpart['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hpart['lk'][level][block], hpart['lk'][level][block+1]
            A_level += [ B_level[r1:r2] @ C_level[r1:r2].T ]
        return block_diag(*A_level)
    

    def shape(self):
        return (self.F.shape[0], self.F.shape[0])


    def _compute_perm_symm_hat_A(self, F:np.ndarray, D:np.ndarray, hpart:mf.EntryHpartDict , ranks:np.ndarray):
        """
        Compute permuted hat_A with each Sigma_level being block diagonal matrix 
        """
        num_levels = ranks.size
        perm_hat_A = np.diag(D)
        for level in range(num_levels - 1):
            perm_hat_A += self._block_diag_FFt(level, hpart, F[:,ranks[:level].sum():ranks[:level+1].sum()])
        return perm_hat_A

    
    def _block_diag_FFt(self, level:int, hpart:mf.EntryHpartDict, F_level:np.ndarray):
        Sigma_level = []
        num_blocks = len(hpart['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hpart['lk'][level][block], hpart['lk'][level][block+1]
            Sigma_level += [ F_level[r1:r2] @ F_level[r1:r2].T ]
        return block_diag(*Sigma_level)
    

    def solve(self, v, eps=1e-9, max_iter=2, printing=False):
        # Solve linear system \Sigma x = v
        if self.inv_B is None: 
            self.inv_coefficients(eps=eps, max_iter=max_iter, printing=printing)

        x = mlr_matvec(v, self.inv_B, self.inv_C, self.inv_hpart, self.inv_ranks)
        v_norm = np.linalg.norm(v)
        assert x.shape == v.shape
        residual = np.linalg.norm(self.matvec(x) - v)
        if printing and residual / v_norm > eps:
            print(f"terminated with {residual/v_norm=}")
        return x
        

    def inv_coefficients(self, refine=True, eps=1e-12, max_iter=1, printing=False, si_groups=None, row_selectors=None):
        prev_l_recurrence = (1/self.D[:, np.newaxis]) * self.F
        n = self.F.shape[0]
        H_Lm1 = np.zeros(self.F.shape)
        L = len(self.hpart['lk']) + 1
        for level in reversed(range(0, L-1)):
            pl = self.hpart['lk'][level].size - 1
            rl = self.ranks[level]
            # M0 same sparsity as Fl
            M0 = prev_l_recurrence[:, -self.ranks[level]:]
            # M1 = M0.T @ rec_term, same sparsity as rec_term
            M1 = np.zeros((rl * pl, self.ranks[:level].sum()))
            for lp in range(level):
                M1[:, self.ranks[:lp].sum() : self.ranks[:lp+1].sum()] = mult_blockdiag_refined_AtB(M0, 
                                                                                    self.hpart['lk'][level], 
                                                                                    self.F[:, self.ranks[:lp].sum():self.ranks[:lp+1].sum()], 
                                                                                    self.hpart['lk'][lp])
            M1_lks = [np.searchsorted(self.hpart['lk'][level], lk_B, side='left') * rl for lk_B in self.hpart['lk'][:level]]
            # M2 = (I + Fl^T M0)^{-1}, blockdiagonal with pl blocks of size (rl x rl)
            FlTM0 = mult_blockdiag_refined_AtB(self.F[:, self.ranks[:level].sum() : self.ranks[:level+1].sum()], 
                                            self.hpart['lk'][level], 
                                            M0, 
                                            self.hpart['lk'][level])
            M2 = np.zeros((pl*rl, rl))
            sqrt_M2 = np.zeros((pl*rl, rl))
            for k in range(pl):
                np.fill_diagonal(FlTM0[k*rl : (k+1)*rl], FlTM0[k*rl : (k+1)*rl].diagonal() + 1)
                eigvals, eigvecs = np.linalg.eigh(FlTM0[k*rl : (k+1)*rl])
                sqrt_M2[k*rl : (k+1)*rl] = ((1 / np.sqrt(eigvals)) * eigvecs) @ eigvecs.T
                M2[k*rl : (k+1)*rl] = ((1/eigvals) * eigvecs) @ eigvecs.T
                del eigvals, eigvecs
            del FlTM0
            H_Lm1[:, self.ranks[:level].sum():self.ranks[:level+1].sum()] = mult_blockdiag_refined_AB(M0, 
                                                                                            self.hpart['lk'][level],
                                                                                            sqrt_M2, 
                                                                                            np.linspace(0, pl*rl, num=pl+1, endpoint=True, dtype=int))
            del sqrt_M2
            # M3 = M2 @ M1, same sparsity as M1
            M3 = np.zeros((rl * pl, self.ranks[:level].sum()))
            for lp in range(level):
                M3[:, self.ranks[:lp].sum():self.ranks[:lp+1].sum()] = mult_blockdiag_refined_AtB(M2, 
                                                                                            np.linspace(0, pl*rl, num=pl+1, endpoint=True, dtype=int), 
                                                                                            M1[:, self.ranks[:lp].sum():self.ranks[:lp+1].sum()], 
                                                                                            M1_lks[lp])
            del M1, M2
            # M4 = M0 @ M3, same sparsity as current rec_term
            M4 = np.zeros((n, self.ranks[:level].sum()))
            for lp in range(level):
                M4[:, self.ranks[:lp].sum() : self.ranks[:lp+1].sum()] = mult_blockdiag_refined_AB(M0, 
                                                                                            self.hpart["lk"][level], 
                                                                                            M3[:, self.ranks[:lp].sum():self.ranks[:lp+1].sum()], 
                                                                                            M1_lks[lp])
            del M0, M3
            # M5  
            prev_l_recurrence = prev_l_recurrence[:, :self.ranks[:level].sum()] - M4
            del M4

        self.inv_B = np.concatenate([-H_Lm1, 1/np.sqrt(self.D).reshape(-1, 1)], axis=1)
        self.inv_C = np.concatenate([H_Lm1, 1/np.sqrt(self.D).reshape(-1, 1)], axis=1)

        if self.si_groups is None:
            self.inv_hpart = {"pi":self.hpart["pi"], "pi_inv":self.hpart["pi_inv"],
                                "lk":self.hpart["lk"] + [np.arange(n+1, dtype=int)]}
            self.row_selectors, self.si_groups, _ = si_row_col(self.hpart)
            self.inv_ranks = self.ranks

        if refine:
            # iterative refinement
            self.inv_B, self.inv_C, self.inv_ranks = iterative_refinement(self.F, H_Lm1, self.D, self.hpart, self.inv_hpart, 
                                                                          self.ranks, self.si_groups, self.row_selectors, 
                                                                          eps=eps, max_iter=max_iter, printing=printing)
            
