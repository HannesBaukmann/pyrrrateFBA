#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:34:51 2020

@author: markukob

QUICK-AND-DIRTY Collection of Linear Algebra Wrappers to have an easier handling of
NUMPY vs. SCIPY-SPARSE Matrices
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def solve_if_unique(amat, bvec):
    """
    solve a linear system amat * xvec == bvec or output None if a unique solution
    cannot be found
    """
    eps = np.finfo(float).eps
    # NUMPY __________________________________________________________________
    if isinstance(amat, np.ndarray):
        if amat.shape[0] < amat.shape[1]: # case 1: underdetermined system
            return None
        if amat.shape[0] == amat.shape[1]: # case 2: square system
            try:
                return np.solve(amat, bvec)
            except:
                return None
        # case 3: potentially over- or underdetermined
        lsqout = np.linalg.lstsq(amat, bvec, rcond=None)
        # x, residual, rank
        if lsqout[2] < amat.shape[0]: # underdetermined
            return None
        if np.linalg.norm(lsqout[1]) > eps*amat.size: # infeasible
            return None
        return lsqout[0]
    # SPARSE _________________________________________________________________
    if isinstance(amat, sp.csr.csr_matrix):
        if amat.shape[0] < amat.shape[1]: # underdetermined
            return None
        if amat.shape[0] == amat.shape[1]: # square
            try:
                return splinalg.spsolve(amat, bvec)
            except:
                return None
        # least squares problem
        lsqout = sp.linalg.lsqr(amat, bvec)
        # 0  1      2    3       4       5      6
        # x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
        if (np.linalg.norm(lsqout[3]) > eps*amat.size) or (lsqout[6] > 1/eps):
            return None
        return lsqout[0]
    raise TypeError('Coefficient matrix must be a numpy nd array or a csr sparse matrix')
