"""
Test the OC function cp_linprog using the model from
10.1016/j.jtbi.2014.10.035
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from ..optimization.oc import cp_linprog
from ..optimization import lp as lp_wrapper

def build_WOB_example():
    """
    Create simple model data for the small reaction chain metabolic network
    """
    alpha = 2.0
    Kkat = 1.0
    P0 = 0.01

    Phi1 = np.array([0.0, -1.0])
    Phi2 = np.array([0.0, 0.0])
    Phi3 = np.array([0.0, 0.0])

    S1 = sp.csr_matrix(np.array([[1.0, -alpha]]))

    S2 = sp.csr_matrix(np.array([[-1.0, 0.0], [0.0, 1.0]]))

    lb = np.array([0.0, 0.0])
    ub = np.array(2*[lp_wrapper.INFINITY])

    h = np.array([0.0])
    Hy = sp.csr_matrix(np.array([[0.0, -1/Kkat]]))
    Hu = sp.csr_matrix(np.array([[1.0, 0.0]]))

    By0 = sp.csr_matrix(np.eye(2, dtype=float))
    Byend = sp.csr_matrix(np.zeros((2, 2), dtype=float))
    b_bndry = np.array([5.0, P0])

    return Phi1, Phi2, Phi3, S1, S2, lb, ub, h, Hy, Hu, By0, Byend, b_bndry


def run_WOB_example():
    """
    Example call
    """
    Phi1, Phi2, Phi3, S1, S2, lb, ub, h, Hy, Hu, By0, Byend, b_bndry = build_WOB_example()

    t_0 = 0.0
    t_end = 10.0
    N = 201

    tt, tt_shift, sol_y, sol_u = cp_linprog(t_0, t_end, Phi1, Phi2, Phi3, S1,
                                            S2, lb, ub, h, Hy, Hu, By0,
                                            Byend, b_bndry, N=N, varphi=0.01)

    plt.subplot(2, 1, 1)
    plt.plot(tt, sol_y)
    plt.subplot(2, 1, 2)
    plt.plot(tt_shift, sol_u)
    plt.show()
