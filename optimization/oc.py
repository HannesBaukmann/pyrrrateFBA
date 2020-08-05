"""
Optimal Control stuff
"""
import numpy as np
import scipy.sparse as sp
from . import lp as lp_wrapper


# TODO: Change at least order of arguments, add additive vector f_1 in qssa rows
def mi_cp_linprog(matrices, t_0, t_end, n_steps=101, varphi=0.0):
    """
    Approximate the solution of the mixed integer optimal control problem
     min int_{t_0}^{t_end} phi1^T y dt + phi2^T y_0 + phi3^T y_end
     s.t.                             y' == smat2*u + smat4*y
                                       0 == smat1*u + smat3*y + f_1
                                   lbvec <= u <= ubvec
                       hmaty*y + hmatu*u <= hvec
                                       0 <= y
          hbmaty*y + hbmatu*u + hbmatx*x <= hbvec
             bmaty0*y_0 + bmatyend*y_end == b_bndry
                                       y in R^{n_y}, u in R^{n_u},
                                       x in B^{n_x}
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


    n_y, n_u = matrices.smat2.shape
    n_x = matrices.matrix_B_x.shape[1]
    n_qssa = matrices.smat1.shape[0]
    n_ally = (n_steps + 1) * n_y
    n_allu = n_steps * n_u
    n_allx = n_steps * n_x
    n_bndry = len(matrices.vec_bndry)

    tgrid = np.linspace(t_0, t_end, n_steps + 1)
    del_t = tgrid[1] - tgrid[0]
    tt_s = (tgrid[1:] + tgrid[:-1]) / 2.0  # time grid for controls

    # Discretization of objective
    # Lagrange part @MAYBE: add possib. for  more complicated objective
    f_y = np.hstack([0.5 * del_t * matrices.phi1,
                     np.hstack((n_steps - 1) * [del_t * matrices.phi1]),
                     0.5 * del_t * matrices.phi1])
    expvals = np.exp(-varphi * tgrid)
    f_y *= np.repeat(expvals, n_y)
    f_u = np.array(n_allu * [0.0])
    f_x = np.array(n_allx * [0.0])
    # Mayer part
    f_y[0:n_y] += matrices.phi2
    f_y[n_steps * n_y:n_ally] += matrices.phi3

    # Discretization of dynamics
    (aeqmat1_y, aeqmat1_u, beq1) = \
        _inflate_constraints(-sp.eye(n_y) + 0.5 * del_t * matrices.smat4, sp.eye(n_y) + 0.5 * del_t * matrices.smat4,
                             del_t * matrices.smat2, np.array(n_y * [[0.0]]), n_steps=n_steps)

    # Discretization of QSSA rows (this is simplified and only works for constant smat1)
    (aeqmat2_y, aeqmat2_u, beq2) = \
        _inflate_constraints(-0.5 * matrices.smat3, 0.5 * matrices.smat3, matrices.smat1, n_qssa * [[0.0]], n_steps=n_steps)

    # Discretization of flux bounds @MAYBE: allow time dependency here
    lb_u = np.vstack(n_steps*[matrices.lbvec])
    ub_u = np.vstack(n_steps*[matrices.ubvec])

    # Discretization of positivity
    lb_y = np.array(n_ally*[[0.0]])
    ub_y = np.array(n_ally*[[lp_wrapper.INFINITY]])

    # Discretization of mixed constraints, This only works for constant smat2
    # TODO: Allow time dependency here
    (amat1_y, amat1_u, bineq1) = _inflate_constraints(0.5*matrices.matrix_y, 0.5*matrices.matrix_y, matrices.matrix_u,
                                                      matrices.vec_h, n_steps=n_steps)

    # Discretization of mixed Boolean constraints
    (amat2_y, amat2_u, bineq2) = _inflate_constraints(0.5*matrices.matrix_B_y, 0.5*matrices.matrix_B_y, matrices.matrix_B_u,
                                                      matrices.vec_B, n_steps=n_steps)
    amat2_x = sp.kron(sp.eye(n_steps), matrices.matrix_B_x)
    # TODO: Use some "_inflate"-mechanism

    # Discretization of equality boundary constraints @MAYBE: also inequality
    aeqmat3_y = sp.hstack([matrices.matrix_start, sp.csr_matrix((n_bndry, (n_steps-1)*n_y)), matrices.matrix_end])
    aeqmat3_u = sp.csr_matrix((n_bndry, n_allu))
    beq3 = matrices.vec_bndry

    # Collect all data
    f_all = np.hstack([f_y, f_u])
    fbar_all = f_x

    aeqmat = sp.bmat([[aeqmat1_y, aeqmat1_u],
                      [aeqmat2_y, aeqmat2_u],
                      [aeqmat3_y, aeqmat3_u]], format='csr')

    beq = np.vstack([beq1, beq2, beq3])
    lb_all = np.vstack([lb_y, lb_u])
    ub_all = np.vstack([ub_y, ub_u])

    amat = sp.bmat([[amat1_y, amat1_u], [amat2_y, amat2_u]], format='csr')
    abarmat = sp.bmat([[sp.csr_matrix((bineq1.shape[0], n_allx))], [amat2_x]])

    bineq = np.vstack([bineq1, bineq2])

    # TODO: Create variable name creator function
    variable_names = ["y_"+str(j+1)+"_"+str(i) for i in range(n_steps+1) for j in range(n_y)]
    variable_names += ["u_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_u)]
    variable_names += ["x_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_x)]

    model = lp_wrapper.MILPModel(name="MIOC Model - Full par., midpoint rule")
    # TODO: Name should be derived from biological model
    model.sparse_mip_model_setup(f_all, fbar_all, amat, abarmat, bineq, aeqmat,
                                 beq, lb_all, ub_all, variable_names)

    model.optimize()

    if model.status == lp_wrapper.OPTIMAL:
        y_data = np.reshape(model.get_solution()[:n_ally], (n_steps+1, n_y))
        u_data = np.reshape(model.get_solution()[n_ally:n_ally+n_allu], (n_steps, n_u))
        x_data = np.reshape(model.get_solution()[n_ally+n_allu:], (n_steps, n_x))
        return tgrid, tt_s, y_data, u_data, x_data
    # TODO: use cleverer output here
    print("No solution found")

    return None



def _inflate_constraints(amat, bmat, cmat, dvec, n_steps=1):
    """
    Create(MI)LP matrix rows from a set of constraints defined on the level of
    the underlying dynamics:
    Assume that for m = 0,1,...,n_steps-1, the following inequalities/equalities
    are given: amat*y_{m+1} + bmat*y_{m} + cmat*u_{m+1/2} <relation> dvec
    """
    amat_y = sp.kron(sp.eye(n_steps, n_steps+1), bmat) + \
             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps+1)), amat)
    amat_u = sp.kron(sp.eye(n_steps), cmat)
    dvec_all = np.vstack(n_steps*[dvec])

    return (amat_y, amat_u, dvec_all)


def _inflate_constraints_new(amat, ttvec, bmat=None):
    """
    Stagging the pointwise given constraints
        amat(tt_m)*y_m + bmat(tt_{m+1})*y_{m+1}
    to a constraint matrix:
       / amat(tt[0]), bmat(tt[1])                                          \
       |             amat(tt[1]), bmat(tt[2])                              |
       |                ...          ...                                   |
       \                                         amat(tt[-2]) bmat(tt[-1]) /
    where tt and N correspond to ttvec and n_tt, resp.

    TODO: Still no irregular grid possible -> replace by generalized Kronecker?
    """
    skip_bmat = True
    n_tt = len(ttvec)
    if callable(amat):
        amat_is_fun = True
        amat0 = amat(ttvec[0])
    else:
        amat_is_fun = False
        amat0 = amat
    n_1, n_2 = amat0.shape
    if not bmat is None:
        skip_bmat = False
        if callable(bmat):
            bmat_is_fun = True
            bmat0 = bmat(ttvec[1])
        else:
            bmat_is_fun = False
            bmat0 = bmat
        n_3, n_4 = bmat0.shape
        if n_1 != n_3 or n_2 != n_4:
            raise Warning("Matrix dimensions are not correct for inflating constraint")
    # build matrices
    if amat_is_fun:
        data, indices, indptr = _inflate_callable(amat, ttvec[:-1], amat0=amat0)
        out_mat = sp.csr_matrix((data, indices, indptr), shape=(n_1*(n_tt-1), n_2*n_tt))
    else:
        out_mat = sp.kron(sp.eye(n_tt-1, n_tt), amat)
    # add bmat-part
    if not skip_bmat:
        if bmat_is_fun:
            data, indices, indptr = _inflate_callable(bmat, ttvec[1:], amat0=bmat0)
            indices = [k+n_2 for k in indices]
            out_mat += sp.csr_matrix((data, indices, indptr), shape=(n_1*(n_tt-1), n_2*n_tt))
        else:
            out_mat += sp.kron(sp.diags([1.0], 1, shape=(n_tt-1, n_tt)), bmat)
    return out_mat


def _inflate_callable(amat, ttvec, **kwargs):
    """
    inflate a given matrix-valued function along the main diagonal of a csr_matrix:
        / amat(ttvec[0])                                      \
       |                  amat(ttvec[1])                      |
       |                      ...                             |
       \                                      amat(ttvec[-1]) /
    Parameters
    ----------
    amat : callable: real -> scipy.sparse.csr_matrix
        function to return amat(t), must have equal shape for all arguments
    ttvec : np.array
        vector of time points
    **kwargs : -"amat0": np.array (2d)
        provide already the first evaluated instance of amat(t)

    Returns
    -------
    data, indices, indptr : arrays for building a csr_matrix
        works as the standard csr_matrix constructor
    """
    n_tt = len(ttvec)
    amat0 = kwargs.get("amat0", amat(ttvec[0]))
    n_2 = amat0.shape[1]
    #nnz = amat0.count_nonzero()
    #n_all = n_tt*nnz #  @MAYBE: a bit larger for safety?
    #data = np.array(n_all*[0.0], dtype=np.float64)
    #indices = np.array(n_all*[0], dtype=np.int32)
    #indptr = np.array(n_tt*n_2*[0], dtype=np.int32)
    data = list(amat0.data)
    indices = list(amat0.indices)
    indptr = list(amat0.indptr[:])
    for i in range(n_tt-1):
        print("i = ", i)
        amat0 = amat(ttvec[i+1])
        data.extend(list(amat0.data))
        indices.extend([n_2*(i+1)+k for k in list(amat0.indices)])
        end = indptr[-1]
        indptr.extend([end + k for k in amat0.indptr[1:]])
    return data, indices, indptr


def _inflate_vec(fvec, ttvec):
    """
    stack possibly time-dependent vectors on top of each other
        ( fvec(ttvec[0], fvec(ttvec[1]), ..., fvec(ttvec[-1])) )
    Parameters
    ----------
    fvec :  np.array
            OR:
            callable double -> np.array of equal length
        vector( function) to be stacked
    ttvec : np.array
        vector of time grid points

    Returns
    -------
    fvec_all: np.array
        stacked vectors
    """
    n_tt = len(ttvec)
    if callable(fvec):
        fvec0 = fvec(ttvec[0])
    else:
        fvec0 = fvec
    n_f = len(fvec0)
    if callable(fvec):
        fvec_all = np.array(n_tt*n_f*[0.0])
        fvec_all[:n_f] = fvec0
        for i in range(n_tt-1):
            fvec_all[(i+1)*n_f:i*n_f] = fvec(ttvec[i+1])
    else:
        fvec_all = np.hstack(n_tt*[fvec])
    return fvec_all


#def _inflate_more_constraints(amat, bmat, cmat, dmat, emat, fvec, n_steps=1):
#    """
#    amat*y_{m+1} + bmat*y_{m} + cmat*u_{m+1/2} + dmat*x_{m+1} + emat*x_{m} <relation> fvec
#    """
#
#    amat_y = sp.kron(sp.eye(n_steps, n_steps + 1), bmat) + \
#             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps + 1)), amat)
#    amat_u = sp.kron(sp.eye(n_steps), cmat)
#    amat_x = sp.kron(sp.eye(n_steps, n_steps + 1), emat) + \
#             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps + 1)), dmat)
#    fvec_all = np.hstack(n_steps * [fvec])
#
#    return (amat_y, amat_u, amat_x, fvec_all)
