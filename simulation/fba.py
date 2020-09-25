"""
Simple flux balance analysis
"""

import numpy as np
from scipy.sparse import csr_matrix
from .results import Solutions
from .. import matrrrices as mat
from ..optimization.lp import LPModel
from ..optimization.oc import mi_cp_linprog


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
            # TODO create Error/error message handler
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
    run_rdeFBA = kwargs.get("run_rdeFBA", True) # TODO Check whether it is possible at all first(!?) or obtain automatically from model
    t_0 = kwargs.get("t_0", 0.0)
    t_end = kwargs.get("t_end", 1.0)
    n_steps = kwargs.get("n_steps", 51)
    varphi = kwargs.get("varphi", 0.0)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    # Call the OC routine
    tt, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(mtx, t_0, t_end, n_steps=n_steps, varphi=varphi)
    
    sols = Solutions(tt, tt_shift, sol_y, sol_u, sol_x)
    
    return sols