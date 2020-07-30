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
    # Molecular / Objective Weights
    nQ = 300.0
    nR = 7459.0
    nT1 = 400.0
    nT2 = 1500.0
    nRP = 300.0
    w = 100.0 # weight of "amino acid" M
    
    # Quota 
    PhiQ = 0.35

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
    kdRP = 0.1

    # Regulation Parameters
    epsilon_RP = 0.01
    epsilon_T2 = 0.01
    epsilon_T2 = 0.001 # WHY ????????????????????????????????
    epsilon_jump = 10.0**-8.0 # small positive number
    alpha = 0.03
    gamma = 20

    # Objective parameters
    new_small_eps = 0.0
    Phi1 = np.array([new_small_eps, new_small_eps, -nQ, -nR, -nT1, -nT2, -nRP])
    Phi2 = np.zeros(7, dtype=float)
    Phi3 = np.zeros(7, dtype=float)

    l = -10.0 ** 8
    u = 10.0 ** 8

    # QSSA matrix (only involving metabolite M)
    S1 = sp.csr_matrix(np.array([[1.0, 1.0, -nQ, -nR, -nT1, -nT2, -nRP]]))

    # probably obsolete matrix coupling y's in the algebraic constraints
    S3 = sp.csr_matrix(np.zeros(7, dtype=float))
    # stoichiometric matrix (I don't like a np.diag here...)
    S2 = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    # damping (degradation) matrix
    S4 = sp.csr_matrix(-np.diag([0.0, 0.0, kdQ, kdR, kdT1, kdT2, kdRP]))

    lb = np.array(7 * [0.0])
    ub = np.array(7 * [INFINITY])

    Hy = sp.csr_matrix(np.array([[0.0, 0.0, (PhiQ - 1) * nQ, PhiQ * nR, PhiQ * nT1, PhiQ * nT2, PhiQ * nRP],
                                 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))

    Hu = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [1.0 / kC1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0 / kC2, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0 / kQ, 1.0 / kR, 1.0 / kT1, 1.0 / kT2, 1.0 / kRP]]))

    h = np.zeros(4, dtype=float)

    HBy = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0
                                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 1
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # 2
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # 3
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 4
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 5
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 6
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])) # 7

    HBu = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))

    HBx = sp.csr_matrix(np.array([[-l, 0.0],
                                  [-(epsilon_jump + u), 0.0],
                                  [0.0, l],
                                  [0.0, epsilon_jump + u],
                                  [epsilon_RP, 0.0],
                                  [-u, 0.0],
                                  [0.0, epsilon_T2],
                                  [0.0, -u]]))

    hB = np.array([-gamma - l,
                   gamma - epsilon_jump,
                   -alpha,
                   alpha + u,
                   0.0, 0.0, 0.0, 0.0])

    bool_con_ind = [0,1,2,3,4,5,6,7]
    #bool_con_ind = [0,1,2,3,4,5,7] # remove v_T2 >= eps_T2*(1-T2bar)
    #bool_con_ind = []
    HBy = HBy[bool_con_ind,:]
    HBu = HBu[bool_con_ind,:]
    HBx = HBx[bool_con_ind,:]
    hB = hB[bool_con_ind]

    con_ind = [0,1,2,3]
    #con_ind = [1,2,3] # remove quota constraint
    #Hu[3,5] = 0.0 # remove enzyme capacity constraint on vT2 only
    #con_ind = []
    Hu = Hu[con_ind,:]
    Hy = Hy[con_ind,:]
    h = h[con_ind]
    
    By0 = sp.csr_matrix(np.eye(7, dtype=float))
    Byend = sp.csr_matrix(np.zeros((7, 7), dtype=float))
    #                   C1     C2     Q      R     T1     T2     RP
    #                   0      1      2      3     4      5      6
    b_bndry = np.array([500.0, 1000.0, 0.15, 0.01, 0.001, 0.001, 0.0])
    #b_bndry[6] = alpha   # RP = alpha -> T2bar = 0
    #b_bndry[2] = 100.0#*b_bndry[2] # ensure quota the hard way
    #b_bndry[3] = 5.0*b_bndry[3] # more R
    #b_bndry[4] = 10.0*b_bndry[4] # more T1 to ensure inflow of carbon source
    #b_bndry[5] = 100.0*b_bndry[5]

    return Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub, h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry


def run_SR_example():
    """
    Example call
    """
    Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub, h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry = build_SR_example()

    t_0 = 0.0
    t_end = 50.0 #0.000001
    N = 101

    tt, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(t_0, t_end, Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub,
                                                      h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry,
                                                      n_steps=N, varphi=0.0001)
    
    plt.subplot(4, 1, 1)
    #            C1 C2 Q, R, T1, T2, RP
    use_y_ind = [0, 1, 2, 3, 4,  5,  6]
    #use_y_ind = [6]
    plt.plot(tt, sol_y[:,use_y_ind])
    #plt.plot(tt, sol_y/(np.outer(np.ones((N+1,1)),abs(sol_y).max(0))))
    plt.subplot(4, 1, 2)
    #            C1 C2 Q  R  T1  T2  RP
    use_u_ind = [0, 1, 2, 3, 4,  5,  6]
    use_u_ind = [5,6]
    
    #plt.plot(tt_shift, sol_u/(np.outer(np.ones((N,1)),abs(sol_u).max(0))))
    #plt.plot(tt_shift, sol_u[:,use_u_ind])
    plt.semilogy(tt_shift, sol_u[:,use_u_ind])
    plt.subplot(4, 1, 3)
    plt.plot(tt_shift, sol_x)

    all_con = []
    for i in range(N):
    #    all_con.append( -(Hy.dot(0.5*(sol_y[i,:]+sol_y[i+1,:])) + Hu.dot(sol_u[i,:]) - h) )
        all_con.append( -(HBy.dot(0.5*(sol_y[i,:]+sol_y[i+1,:])) + HBx.dot(sol_x[i,:]) + HBu.dot(sol_u[i,:]) - hB))
    plt.subplot(4,1,4)
    plt.semilogy(tt_shift, np.array(all_con))
    plt.legend(["1","2","3","4","6","8"])
    plt.show()
