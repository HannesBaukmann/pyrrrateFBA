"""
Test the OC function milp_cp_linprog using the SR model from Lin's paper
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from ..optimization.oc import mi_cp_linprog
from ..optimization.lp import INFINITY
#from ..optimization import lp as lp_wrapper

def build_SR_example():
    """
    Create Regulatory Self-Replicator model data
    """
    # Turnover rates
    kC1 = 3000
    kC2 = 2000
    kQ = 4.2
    kR = 0.1689
    kT1 = 3.15
    kT2 = 0.81
    kRP = 4.2

    # Degradation rates
    kdQ = 0.01
    kdR = 0.01
    kdT1 = 0.01
    kdT2 = 0.01
    kdRP = 0.2

    # Regulation Parameters
    epsilon_RP = 0.01
    epsilon_T2 = 0.01
    epsilon_jump = 0.001  # small positive number
    alpha = 0.03
    gamma = 20

    # Objective parameters
    Phi1 = np.array([0.0, 0.0, -300.0, -7459.0, -400.0, -1500.0, -300.0])
    Phi2 = np.zeros(7, dtype=float)
    Phi3 = np.zeros(7, dtype=float)

    l = -INFINITY
    u = INFINITY

    #
    S1 = sp.csr_matrix(np.array([[1.0, 1.0, -300.0, -7459.0, -400.0, -1500.0, -300.0]]))

    S3 = sp.csr_matrix(np.zeros(7, dtype=float))

    S2 = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

    S4 = sp.csr_matrix(-np.diag([0.0, 0.0, kdQ, kdR, kdT1, kdT2, kdRP]))
    #S4 = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                             [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #                             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    #                             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

    lb = np.array(7 * [0.0])
    ub = np.array(7 * [INFINITY])

    # Warum nutzen wir hier nicht Phi_Q?
    Hy = sp.csr_matrix(np.array([[0.0, 0.0, -195.0, 2610.65, 140.0, 525.0, 105.0],
                                 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))

    Hu = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [1.0 / kC1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0 / kC2, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0 / kQ, 1.0 / kR, 1.0 / kT1, 1.0 / kT2, 1.0 / kRP]]))

    h = np.zeros(4, dtype=float)

    HBy = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    HBu = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))

    HBx = sp.csr_matrix(np.array([[-l, 0.0],
                                  [epsilon_jump - u, 0.0],
                                  [0.0, -l],
                                  [0.0, epsilon_jump - u],
                                  [epsilon_RP, 0.0],
                                  [-u, 0.0],
                                  [0.0, epsilon_T2],
                                  [0.0, -u]]))

    hB = np.array([gamma - l, gamma - epsilon_jump, alpha - l, alpha - epsilon_jump, 0.0, 0.0, 0.0, 0.0])

    By0 = sp.csr_matrix(np.eye(7, dtype=float))
    Byend = sp.csr_matrix(np.zeros((7, 7), dtype=float))
    b_bndry = np.array([1000.0, 500.0, 0.15, 0.01, 0.001, 0.001, 0.0])

    return Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub, h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry


def run_SR_example():
    """
    Example call
    """
    Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub, h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry = build_SR_example()

    t_0 = 0.0
    t_end = 50.0
    N = 201

    tt, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(t_0, t_end, Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub, h, Hy, Hu, By0,
                                            HBy, HBu, HBx, hB, Byend, b_bndry, n_steps=N, varphi=0.01)

    plt.subplot(3, 1, 1)
    plt.plot(tt, sol_y)
    plt.subplot(3, 1, 2)
    plt.plot(tt_shift, sol_u)
    plt.subplot(3, 1, 3)
    plt.plot(tt_shift, sol_x)
    plt.show()
