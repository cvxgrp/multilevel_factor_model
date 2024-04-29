import copy
from typing import List, Tuple, Callable, TypedDict, List, Set, Optional 

import numpy as np
from scipy.linalg import block_diag

import mlrfit as mf


from mfmodel.utils import *
from mfmodel.inverse import *
from mfmodel.fast_em_algorithm import *




"""
Multilevel Factor Model class
"""
class MFModel:
    def __init__(self, hpart:Optional[mf.EntryHpartDict ]=None, ranks:Optional[np.ndarray]=None, \
                       F:Optional[np.ndarray]=None, D:Optional[np.ndarray]=None, debug=False):
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
        return F, D
    

    def diag(self):
        # return diagonal of Sigma
        return (self.diag_sparse_FFt(self.F, self.hpart, self.ranks) + self.D)[self.pi_inv]


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
    

    def solve(self, v, eps=1e-9, max_iter=20, printing=False):
        # Solve linear system \Sigma x = v
        return iterative_refinement(self.ranks, v, self.F, self.D, self.hpart, 
                                    eps=eps, max_iter=max_iter, printing=printing)
    
    
    def num_factors(self):
        # return number of unique factors
        return (np.array([self.hpart["lk"][l].size-1 for l in range(len(self.hpart["lk"]))]) * self.ranks[:-1]).sum()
    
    
    def matrix(self):
        # return \Sigma matrix
        perm_hat_A = self._compute_perm_hat_A(self.F, self.D, self.hpart, self.ranks)
        # pi_inv to permute \hat \Sigma_l from block diagonal in order approximating \Sigma
        hat_A = perm_hat_A[self.pi_inv, :][:, self.pi_inv]
        return hat_A
    

    def shape(self):
        return (self.F.shape[0], self.F.shape[0])


    def _compute_perm_hat_A(self, F:np.ndarray, D:np.ndarray, hpart:mf.EntryHpartDict , ranks:np.ndarray):
        """
        Compute permuted hat_A with each Sigma_level being block diagonal matrix 
        """
        num_levels = ranks.size
        perm_hat_A = np.diag(D)
        for level in range(num_levels - 1):
            perm_hat_A += self._block_diag_FFt(level, hpart, F[:,ranks[:level].sum() : ranks[:level+1].sum()])
        return perm_hat_A

    
    def _block_diag_FFt(self, level:int, hpart:mf.EntryHpartDict, F_level:np.ndarray):
        Sigma_level = []
        num_blocks = len(hpart['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hpart['lk'][level][block], hpart['lk'][level][block+1]
            Sigma_level += [ F_level[r1:r2] @ F_level[r1:r2].T ]
        return block_diag(*Sigma_level)
