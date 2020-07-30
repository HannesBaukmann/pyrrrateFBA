# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:21:02 2020

@author: Markus
"""

import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from ..optimization.oc import cp_linprog


def exp_example():
    """
    Simple test to see whether
    (a) the oc routines work if we insert arguments with length zero
    (b) whether a simple time integration task works
    We solve the ODE y' = - phi1*y, y(0) == 1
    with the solution y(t)= exp(-phi1*t)
    """
    S1 = sp.csr_matrix(np.array((1,0)))
    S2 = sp.csr_matrix(np.array((0,0)))
    phi1 = 2.0
    S4 = sp.csr_matrix(np.array([[-phi1]]))
    S3 = sp.csr_matrix((1,1))

    Phi1 = np.array([1.0])
    Phi2 = np.array([0.0])
    Phi3 = np.array([0.0])

    lb = np.array((0,0))
    ub = np.array((0,0))

    h = np.array([])
    Hy = sp.csr_matrix(np.array((0,1)))
    Hu = sp.csr_matrix(np.array((0,0)))

    By0 = sp.csr_matrix(np.array([[1.0]]))
    Byend = sp.csr_matrix((1,1))
    b_bndry = np.array([1.0])
    N = 201
    t_0 = 0.0
    t_end = 1.0

    tt, tt_shift, sol_y, sol_u = cp_linprog(t_0, t_end, Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub,
                                                          h, Hy, Hu, By0, Byend, b_bndry,
                                                          n_steps=N, varphi=0.0001)
    print(np.abs(sol_y[N] - np.exp(-phi1*t_end)))
    plt.plot(tt, sol_y)
    plt.plot(tt, np.exp(-phi1*tt))
    plt.show()