"""
Simple flux balance analysis
"""

import numpy as np
from scipy.sparse import csr_matrix
from ..simulation.results import Solutions
from .. import matrrrices as mat
from ..optimization.lp import LPModel
from ..optimization.oc import mi_cp_linprog, cp_rk_linprog


def perform_fba(model, **kwargs):
    """
    Classical Flux Balance Analysis by LP solution

    Parameters
    ----------
    model : PYRRRATEFBA Model
        main data structure of the underlying model.
    **kwargs :
        'objective': - None: The biomass reactions will be detected by name.
                     - vector of length #reactions

    Returns
    -------
    TODO
    """
    fvec = kwargs.get("objective", None)
    maximize = kwargs.get("maximize", True)
    #
    # Get QSSA stoichiometric matrix
    smat = model.stoich # FIXME: This is wrong if we have macromolecules in the model,
    # better: choose smat1 from the Matrrrices but then we probably need other flux
    # constraints or have to give an error message
    nrows, ncols = smat.shape
    high_reaction_flux = 1000.0
    lbvec = np.zeros(ncols)  # lower flux bounds
    ubvec = np.zeros(ncols)  # upper flux bounds
    for idx, rxn in enumerate(model.reactions_dict.values()):
        ubvec[idx] = rxn.get('upperFluxBound', high_reaction_flux)
        if rxn['reversible']:
            lbvec[idx] = rxn.get('lowerFluxBound', -high_reaction_flux)
        else:
            lbvec[idx] = rxn.get('lowerFluxBound', 0.0)
    fvec = np.zeros(ncols)

    if fvec is None:
        # biomass reaction indices
        brxns = [idx for idx, reac in enumerate(model.reactions_dict.keys())
                 if 'biomass' in reac.lower()]
        if not brxns:
            print('No biomass reaction found and no objective provided, exiting.')
            return None
        fvec[brxns] = -1.0
    if maximize:
        fvec = -fvec
    # set up the LP
    lp_model = LPModel()
    # fvec, amat, bvec, aeqmat, beq, lbvec, ubvec, variable_names
    lp_model.sparse_model_setup(fvec, csr_matrix((0, ncols)), np.zeros(0),
                                csr_matrix(smat), np.zeros(nrows), lbvec, ubvec,
                                list(model.reactions_dict.keys()))
    lp_model.optimize()
    sol = lp_model.get_solution()
    return sol


def perform_rdefba(model, **kwargs):
    """
    Use (r)deFBA to approximate the dynamic behavior of the model

    Parameters
    ----------
    model : PyrrrateFBAModel
        main biochemical model.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    sol : Solutions instance containing the time series information

    TODO : More options to "play around"
    """
    run_rdeFBA = kwargs.get('run_rdeFBA', True)
    if run_rdeFBA and not model.can_rdeFBA:
        raise ValueError('Cannot run an r-deFBA on this model.')
    t_0 = kwargs.get('t_0', 0.0)
    t_end = kwargs.get('t_end', 1.0)
    n_steps = kwargs.get('n_steps', 51)
    varphi = kwargs.get('varphi', 0.0)
    rkm = kwargs.get('runge_kutta', None)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    # adapt initial values if explicitly given
    y_0 = kwargs.get('set_y0', None)
    if y_0 is not None:
        mtx.matrix_end = csr_matrix((y_0.size, y_0.size))
        mtx.matrix_start = csr_matrix(np.eye(y_0.size))
        mtx.vec_bndry = y_0.transpose()
    # Call the OC routine
    if rkm is None:
        tgrid, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(mtx, t_0, t_end, n_steps=n_steps,
                                                             varphi=varphi)
    else:
        if run_rdeFBA:
            print('Cannot (yet) run r-deFBA with arbitrary Runge-Kutta scheme. Fallback to deFBA')
        tgrid, tt_shift, sol_y, sol_u = cp_rk_linprog(mtx, rkm, t_0, t_end, n_steps=n_steps,
                                                      varphi=varphi)
        #sol_x = np.zeros((0, sol_u.shape[1]))
        sol_x = np.zeros((sol_u.shape[0], 0))

    sols = Solutions(tgrid, tt_shift, sol_y, sol_u, sol_x)

    return sols


def perform_soa_rdeFBA(model, **kwargs):
    """
    iterative process consisting of several (r)deFBA runs with a very crude one-step
    approximation in each step (quasi-_S_tatic _O_ptimization _A_pproach)
    # MAYBE: This could become a special case of short-term (r-)deFBA
    # QUESTION: Can it be a problem if set_y0 is set from the start?
       (quasi-recursive call of the algorithms...)
    """
    run_rdeFBA = kwargs.get('run_rdeFBA', True)
    n_steps = kwargs.get('n_steps', 51)
    tgrid = np.linspace(kwargs.get('t_0', 0.0), kwargs.get('t_end', 1.0), n_steps)
    varphi = kwargs.get('varphi', 0.0)
    kwargs.pop('varphi', kwargs)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    y_0 = mtx.extract_initial_values()
    if y_0 is None:
        print('SOA (r)deFBA cannot be perforrrmed.')
        return None
    kwargs['set_y0'] = y_0
    kwargs['n_steps'] = 1
    #
    tslice = tgrid[0:2]
    sols = model.rdeFBA(tslice, varphi, do_soa=False, **kwargs)
    y_new = np.array(sols.dyndata.tail(n=1)) # row
    for k in range(1, n_steps-1):
        kwargs['set_y0'] = y_new # row
        tslice = tgrid[k:k+2]
        sol_tmp = model.rdeFBA(tslice, varphi, do_soa=False, **kwargs)
        # QUESTION: What if internal error occurs here?
        new_t_shift = sol_tmp.condata.index[-1]
        ux_new = np.array(sol_tmp.condata.tail(n=1))
        y_new = np.array(sol_tmp.dyndata.tail(n=1))
        sols.extend_y([tslice[-1]], y_new)
        sols.extend_ux([new_t_shift], ux_new)
    #print(sols)
    return sols


#def perform_shortterm_rdeFBA(model, **kwargs):
#    """
#    short-term (r-)deFBA
#    """
#    t_0 = kwargs.get('t_0', 0.0)
