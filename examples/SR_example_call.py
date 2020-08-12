"""
Test the OC function milp_cp_linprog using the SR model from Lin's paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from ..optimization.oc import mi_cp_linprog
from ..optimization.lp import INFINITY
#from ..optimization import lp as lp_wrapper

class Solutions:
    def __init__(self, tt, tt_shift, sol_y, sol_u, sol_x): 
        self.tt = tt
        self.tt_shift = tt_shift 
        self.sol_y = sol_y
        self.sol_u = sol_u
        self.sol_x = sol_x

class SR_Matrices:
    def __init__(self):
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
        # kdRP = 0.2
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
        self.phi1 = np.array([new_small_eps, new_small_eps, -nQ, -nR, -nT1, -nT2, -nRP])
        self.phi2 = np.zeros(7, dtype=float)
        self.phi3 = np.zeros(7, dtype=float)
    
        l = -10.0 ** 8
        u = 10.0 ** 8
    
        # QSSA matrix (only involving metabolite M)
        self.smat1 = sp.csr_matrix(np.array([[1.0, 1.0, -nQ, -nR, -nT1, -nT2, -nRP]]))
    
        # probably obsolete matrix coupling y's in the algebraic constraints
        self.smat3 = sp.csr_matrix(np.zeros(7, dtype=float))
        # stoichiometric matrix (I don't like a np.diag here...)
        self.smat2 = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        # damping (degradation) matrix
        self.smat4 = sp.csr_matrix(-np.diag([0.0, 0.0, kdQ, kdR, kdT1, kdT2, kdRP]))
    
        self.lbvec = np.array(7 * [[0.0]])
        self.ubvec = np.array(7 * [[INFINITY]])
    
        self.matrix_y = sp.csr_matrix(np.array([[0.0, 0.0, (PhiQ - 1) * nQ, PhiQ * nR, PhiQ * nT1, PhiQ * nT2, PhiQ * nRP],
                                     [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                     [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))
    
        self.matrix_u = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0 / kC1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0 / kC2, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0 / kQ, 1.0 / kR, 1.0 / kT1, 1.0 / kT2, 1.0 / kRP]]))
    
        self.vec_h = np.array(4 * [[0.0]])
    
        self.matrix_B_y = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0
                                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 1
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # 2
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # 3
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 4
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 5
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 6
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])) # 7
    
        self.matrix_B_u = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
    
        self.matrix_B_x = sp.csr_matrix(np.array([[-l, 0.0],
                                      [-(epsilon_jump + u), 0.0],
                                      [0.0, l],
                                      [0.0, epsilon_jump + u],
                                      [epsilon_RP, 0.0],
                                      [-u, 0.0],
                                      [0.0, epsilon_T2],
                                      [0.0, -u]]))
    
        self.vec_B = np.array([[-gamma - l],
                       [gamma - epsilon_jump],
                       [-alpha],
                       [alpha + u],
                       [0.0], [0.0], [0.0], [0.0]])
#        bool_con_ind = [0,1,2,3,4,5,6,7]
        #bool_con_ind = [0,1,2,3,4,5,7] # remove v_T2 >= eps_T2*(1-T2bar)
        #bool_con_ind = []
#        self.matrix_B_y = self.matrix_B_y[bool_con_ind,:]
#        self.matrix_B_u = self.matrix_B_u[bool_con_ind,:]
#        self.matrix_B_x = self.matrix_B_x[bool_con_ind,:]
#        self.vec_B = self.vec_B[bool_con_ind]
    
#        con_ind = [0,1,2,3]
        #con_ind = [1,2,3] # remove quota constraint
        #Hu[3,5] = 0.0 # remove enzyme capacity constraint on vT2 only
        #con_ind = []
#        self.matrix_u = self.matrix_u[con_ind,:]
#        self.matrix_y = self.matrix_y[con_ind,:]
#        self.vec_h = self.vec_h[con_ind]
        
        self.matrix_start = sp.csr_matrix(np.eye(7, dtype=float))
        self.matrix_end = sp.csr_matrix(np.zeros((7, 7), dtype=float))
        #                   C1     C2     Q      R     T1     T2     RP
        #                   0      1      2      3     4      5      6
        self.vec_bndry = np.array([[500.0], [1000.0], [0.15], [0.01], [0.001], [0.001], [0.0]])
        #b_bndry[6] = alpha   # RP = alpha -> T2bar = 0
        #b_bndry[2] = 100.0#*b_bndry[2] # ensure quota the hard way
        #b_bndry[3] = 5.0*b_bndry[3] # more R
        #b_bndry[4] = 10.0*b_bndry[4] # more T1 to ensure inflow of carbon source
        #b_bndry[5] = 100.0*b_bndry[5]

def run_SR_example():
    sr_mtx = SR_Matrices()
    
    t_0 = 0.0
    t_end = 55.0 #0.000001
    N = 101
        
    tt, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(sr_mtx, t_0, t_end, n_steps=N, varphi=0.001)
        
    sols = Solutions(tt, tt_shift, sol_y, sol_u, sol_x)
    
    # extracellular Carbon species
    ext = pd.DataFrame()
    ext['time'] = sols.tt
    ext['C1'] = sols.sol_y[:,0]
    ext['C2'] = sols.sol_y[:,1]
        
    ext.plot(x='time')
    plt.xlim(0,55)
    plt.xlabel('Time / min')    
    plt.show()
    
    # macromolecules
    mac = pd.DataFrame()
    mac['time'] = sols.tt
    # mac['Q'] = sols.sol_y[:,2]
    mac['R'] = sols.sol_y[:,3]
    mac['T1'] = sols.sol_y[:,4]
    mac['T2'] = sols.sol_y[:,5]
    mac['RP'] = sols.sol_y[:,6]
    
    mac.plot(x='time')
    plt.xlim(0,55)
    plt.xlabel('Time / min')    
    plt.show()

    # qualitative species    
    qual = pd.DataFrame()
    qual['time'] = sols.tt_shift
    qual['RPbar'] = sols.sol_x[:,0]
    qual['T2bar'] = sols.sol_x[:,1]
    
    qual.plot(x='time')
    plt.xlim(0,55)
    plt.xlabel('Time / min')    
    plt.show()
    
    # translation reactions
    translation = pd.DataFrame()
    translation['time'] = sols.tt_shift
    translation['v_Q'] = sols.sol_u[:,2]
    translation['v_R'] = sols.sol_u[:,3]
    translation['v_T1'] = sols.sol_u[:,4]
    translation['v_T2'] = sols.sol_u[:,5]
    translation['v_RP'] = sols.sol_u[:,6]
    
    translation.plot(x='time')
    plt.xlim(0,55)
    plt.xlabel('Time / min')    
    plt.show()
