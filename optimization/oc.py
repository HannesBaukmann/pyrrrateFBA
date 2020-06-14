"""
Optimal Control stuff
"""
import numpy as np
import scipy.sparse as sp
from . import lp as lp_wrapper


def cp_linprog(t_0, t_end, phi1, phi2, phi3, smat1, smat2, lb, ub, h, Hy, Hu,
               By0, Byend, b_bndry, n_steps=101, varphi=0.0):
    """
    Approximate the solution of the optimal control problem
     min int_{t_0}^{t_end} phi1^T y dt + phi2^T y_0 + phi3^T y_end
     s.t. y' = smat2 u
          0  = smat1 u
          lb <= u <= ub
          Hy y + Hu u <= h
          0 <= y
          By0*y_0 + Byend*y_end = b_bndry
    using complete parameterization with N time intervals and midpoint rule for
     DAE integration and trapezoidal rule for approximating the integral in the
     objective
    all vectors are always supposed to be np.array, all matrices
     scipy.sparse.csr_matrix
    @DEBUG
     - This is supposed to be temporary and replaced by a more general oc
       routine
    @TODO
     - create/use a class for the time series data in the output
     - more security checks
     - add additional terms in objectvie and dynamics
     - allow irregular time grid
     - ...
    """
    n_y, n_u = smat2.shape
    n_qssa = smat1.shape[0]
    n_ally = (n_steps+1)*n_y
    n_allu = n_steps*n_u
    n_bndry = len(b_bndry)

    tt = np.linspace(t_0, t_end, n_steps+1)
    del_t = tt[1]-tt[0]
    tt_shift = (tt[1:] + tt[:-1])/2.0 # time grid for controls

    # Discretization of objective
    # Lagrange part @MAYBE: add possib. for  more complicated objective
    f_y = np.hstack([0.5*del_t*phi1,
                     np.hstack((n_steps-1)*[del_t*phi1]),
                     0.5*del_t*phi1])
    expvals = np.exp(-varphi*tt)
    f_y *= np.repeat(expvals, n_y)
    f_u = np.array(n_allu*[0.0])
    # Mayer part
    f_y[0:n_y] += phi2
    f_y[n_steps*n_y:n_ally] += phi3

    # Discretization of dynamics
    Aeq1_y = sp.kron(sp.diags((1.0, -1.0), (0, 1), (n_steps, n_steps+1)),
                     sp.eye(n_y))
    Aeq1_u = sp.kron(sp.eye(n_steps), del_t*smat2)
    beq1 = np.array(n_steps*n_y*[0.0])

    # Discretization of QSSA rows (this is simplified and only works for constant smat1)
    Aeq2_y = sp.csr_matrix((n_steps*n_qssa, n_ally))
    Aeq2_u = sp.kron(sp.eye(n_steps), smat1)
    beq2 = np.array(n_steps*n_qssa*[0.0])

    # Discretization of flux bounds @MAYBE: allow time dependency here
    lb_u = np.hstack(n_steps*[lb])
    ub_u = np.hstack(n_steps*[ub])

    # Discretization of positivity
    lb_y = np.array(n_ally*[0.0])
    ub_y = np.array(n_ally*[lp_wrapper.INFINITY])

    # Discretization of mixed constraints
    # TODO: Allow time dependency here
    Aineq1_y = sp.kron(sp.diags((1.0, 1.0), (0, 1),
                                shape=(n_steps, n_steps+1)), 0.5*Hy)
    Aineq1_u = sp.kron(sp.eye(n_steps), Hu)
    bineq1 = np.hstack(n_steps*[h])

    # Discretization of equality boundary constraints @MAYBE: also inequality
    Aeq3_y = sp.hstack([By0, sp.csr_matrix((n_bndry, (n_steps-1)*n_y)), Byend])
    Aeq3_u = sp.csr_matrix((n_bndry, n_allu))
    beq3 = b_bndry

    # Collect all data
    f = np.hstack([f_y, f_u])
    Aeq = sp.bmat([[Aeq1_y, Aeq1_u],
                   [Aeq2_y, Aeq2_u],
                   [Aeq3_y, Aeq3_u]], format='csr')
    beq = np.hstack([beq1, beq2, beq3])
    lb_all = np.hstack([lb_y, lb_u])
    ub_all = np.hstack([ub_y, ub_u])
    Aineq = sp.bmat([[Aineq1_y, Aineq1_u]], format='csr')
    bineq = bineq1

    # TODO: Create variable name creator function
    variable_names_y = ["y_"+str(j+1)+"_"+str(i) for i in range(n_steps+1) for j in range(n_y)]
    variable_names_u = ["u_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_u)]
    variable_names = variable_names_y + variable_names_u

    Model = lp_wrapper.LPModel(name="OC Model - Full par., midpoint rule")
    Model.sparse_model_setup(f, Aineq, bineq, Aeq, beq, lb_all, ub_all, variable_names)

    Model.optimize()

    if Model.status == lp_wrapper.OPTIMAL: # @FIXME: This is gurobi specific.
        y_data = np.reshape(Model.get_solution()[:n_ally], (n_steps+1, n_y))
        u_data = np.reshape(Model.get_solution()[n_ally:], (n_steps, n_u))
        return tt, tt_shift, y_data, u_data
    else:
        # TODO: use cleverer output here
        print("No solution found")

    return None
