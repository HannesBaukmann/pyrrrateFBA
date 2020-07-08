"""
Optimal Control stuff
"""
import numpy as np
import scipy.sparse as sp
from . import lp as lp_wrapper


def cp_linprog(t_0, t_end, phi1, phi2, phi3, smat1, smat2, lbvec, ubvec, hvec, hmaty, hmatu,
               bmaty0, bmatyend, b_bndry, n_steps=101, varphi=0.0):
    """
    Approximate the solution of the optimal control problem
     min int_{t_0}^{t_end} phi1^T y dt + phi2^T y_0 + phi3^T y_end
     s.t. y' = smat2 u
          0  = smat1 u
          lbvec <= u <= ubvec
          hmaty y + hmatu u <= hvec
          0 <= y
          bmaty0*y_0 + bmatyend*y_end = b_bndry
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
     - add additional terms in objective and dynamics
     - allow irregular time grid
     - ...
    """
    n_y, n_u = smat2.shape
    n_qssa = smat1.shape[0]
    n_ally = (n_steps+1)*n_y
    n_allu = n_steps*n_u
    n_bndry = len(b_bndry)

    tgrid = np.linspace(t_0, t_end, n_steps+1)
    del_t = tgrid[1]-tgrid[0]
    tt_shift = (tgrid[1:] + tgrid[:-1])/2.0 # time grid for controls

    # Discretization of objective
    # Lagrange part @MAYBE: add possib. for  more complicated objective
    f_y = np.hstack([0.5*del_t*phi1,
                     np.hstack((n_steps-1)*[del_t*phi1]),
                     0.5*del_t*phi1])
    expvals = np.exp(-varphi*tgrid)
    f_y *= np.repeat(expvals, n_y)
    f_u = np.array(n_allu*[0.0])
    # Mayer part
    f_y[0:n_y] += phi2
    f_y[n_steps*n_y:n_ally] += phi3

    # Discretization of dynamics
    (aeqmat1_y, aeqmat1_u, beq1) = \
      _inflate_constraints(-sp.eye(n_y), sp.eye(n_y), del_t*smat2,
                           np.array(n_y*[0.0]), n_steps=n_steps)

    # Discretization of QSSA rows (this is simplified and only works for constant smat1)
    (aeqmat2_y, aeqmat2_u, beq2) = \
        _inflate_constraints(sp.csr_matrix((n_qssa, n_y)), sp.csr_matrix((n_qssa, n_y)),
                             smat1, n_qssa*[0.0], n_steps=n_steps)

    # Discretization of flux bounds @MAYBE: allow time dependency here
    lb_u = np.hstack(n_steps*[lbvec])
    ub_u = np.hstack(n_steps*[ubvec])

    # Discretization of positivity
    lb_y = np.array(n_ally*[0.0])
    ub_y = np.array(n_ally*[lp_wrapper.INFINITY])

    # Discretization of mixed constraints, This only works for constant smat2
    # TODO: Allow time dependency here
    (amat1_y, amat1_u, bineq1) = _inflate_constraints(0.5*hmaty, 0.5*hmaty, hmatu,
                                                      hvec, n_steps=n_steps)

    # Discretization of equality boundary constraints @MAYBE: also inequality
    aeqmat3_y = sp.hstack([bmaty0, sp.csr_matrix((n_bndry, (n_steps-1)*n_y)), bmatyend])
    aeqmat3_u = sp.csr_matrix((n_bndry, n_allu))
    beq3 = b_bndry

    # Collect all data
    f_all = np.hstack([f_y, f_u])
    aeqmat = sp.bmat([[aeqmat1_y, aeqmat1_u],
                      [aeqmat2_y, aeqmat2_u],
                      [aeqmat3_y, aeqmat3_u]], format='csr')
    beq = np.hstack([beq1, beq2, beq3])
    lb_all = np.hstack([lb_y, lb_u])
    ub_all = np.hstack([ub_y, ub_u])
    amat = sp.bmat([[amat1_y, amat1_u]], format='csr')
    bineq = bineq1

    # TODO: Create variable name creator function
    variable_names = ["y_"+str(j+1)+"_"+str(i) for i in range(n_steps+1) for j in range(n_y)]
    variable_names += ["u_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_u)]

    model = lp_wrapper.LPModel(name="OC Model - Full par., midpoint rule")
    # TODO: Name should be derived from biological model
    model.sparse_model_setup(f_all, amat, bineq, aeqmat, beq, lb_all, ub_all, variable_names)

    model.optimize()

    if model.status == lp_wrapper.OPTIMAL:
        y_data = np.reshape(model.get_solution()[:n_ally], (n_steps+1, n_y))
        u_data = np.reshape(model.get_solution()[n_ally:], (n_steps, n_u))
        return tgrid, tt_shift, y_data, u_data
    # TODO: use cleverer output here
    print("No solution found")

    return None


def _inflate_constraints(amat, bmat, cmat, dvec, n_steps=1):
    """
    Create(MI)LP matrix rows from a set of constraints defined on the level of
    the underlying dynamics:
    Assume that for m = 0,1,...,n_steps-1, the following inequalities/equalities
    are given: amat*y_{m+1} + bmat*y_{m} + cmat*u_{m+1/2} <relation> dvec
    #TODO: - Allow time dependency
           - include possible internal stages
    """
    amat_y = sp.kron(sp.eye(n_steps, n_steps+1), bmat) + \
             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps+1)), amat)
    amat_u = sp.kron(sp.eye(n_steps), cmat)
    dvec_all = np.hstack(n_steps*[dvec])

    return (amat_y, amat_u, dvec_all)

#def mi_cp_linprog()
