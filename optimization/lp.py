"""
Collection of routines for (Mixed Integer) Linear Programming
"""

import numpy as np
import scipy.sparse as sp

DEFAULT_SOLVER = 'gurobi'
#DEFAULT_SOLVER = 'soplex'
#DEFAULT_SOLVER = 'glpk'
if DEFAULT_SOLVER == 'gurobi':
    try:
        import gurobipy
        # Module constants
        INFINITY = gurobipy.GRB.INFINITY
        OPTIMAL = gurobipy.GRB.OPTIMAL
        MINIMIZE = gurobipy.GRB.MINIMIZE
    except ImportError:
        DEFAULT_SOLVER = 'soplex'
if DEFAULT_SOLVER == 'soplex':
    try:
        import pyscipopt
        # Module constants
        INFINITY = pyscipopt.Model().infinity()
        OPTIMAL = 'optimal' # pyscipopt.SCIP_RESULT.FOUNDSOL -> 15 ???
        MINIMIZE = 'minimize'
    except ImportError:
        DEFAULT_SOLVER = 'glpk'
        # Module constants
        #INFINITY =
        OPTIMAL = 'opt'
        #MINIMIZE =
if DEFAULT_SOLVER == 'glpk':
    try:
        import glpk
    except ImportError:
        DEFAULT_SOLVER = 'scipy'
if DEFAULT_SOLVER == 'scipy':
    try:
        import scipy.optimize as sciopt
        print('Cannot handle integer constraints when using scipy.optimize.linprog')
    except ImportError as err:
        raise(err)


# Constants for BigM and small M constraints
EPSILON = 10**-6
BIGM = 10**8
MINUSBIGM = -BIGM

class LPModel():
    """
    Simple wrapper class to handle various LP solvers in a unified way
    TODO - add more solvers, more options, allow for heuristics
    """
    def __init__(self, name=""):
        self.name = name
        self.solver_name = DEFAULT_SOLVER
        if self.solver_name == 'gurobi':
            self.solver_model = gurobipy.Model()
        if self.solver_name == 'soplex':
            self.solver_model = pyscipopt.Model()
        if self.solver_name == 'glpk':
            self.solver_model = glpk.LPX()
        if self.solver_name == 'scipy':
            self.solver_model = _Scipy_LP_model() # FIXME: Implement!
        self.status = "Unknown"


    def get_solution(self):
        """
        Output the solution vector of the LP if already calculated
        """
        if self.solver_name == 'gurobi':
            if self.status != OPTIMAL:
                return None
            return self.solver_model.x
        elif self.solver_name == 'soplex':
            if self.status != 'optimal':
                return None
            #return np.array([self.solver_model.getVal(x) for x in self.solver_model.getVars()]).transpose()
            #return np.array([self.solver_model.getVal(x) for x in self.solver_model.getVars()]) scip sorts the variables: Booleans first
            return np.array([self.solver_model.getVal(x) for x in self.solver_model.data])
            # MAYBE: find a more elegant and less error-prone way of extracting solutions here
        return None


    def sparse_model_setup(self, fvec, amat, bvec, aeqmat, beq, lbvec, ubvec, variable_names):
        """
        Fill the "solver_model" of the underlying LP solver class with the
        following LP:
            min fvec'*x
            s.t. amat*x <= bvec
                 aeqmat*x == beq
                 lbvec <= x <= ubvec
            and the variable names stored in the list "variable_names"
        """
        if self.solver_name == 'gurobi':
            _sparse_model_setup_gurobi(self.solver_model,
                                       fvec, amat, bvec, aeqmat, beq,
                                       lbvec, ubvec, variable_names)
        elif self.solver_name == 'soplex':
            _sparse_model_setup_soplex(self.solver_model,
                                       fvec, amat, bvec, aeqmat, beq,
                                       lbvec, ubvec, variable_names)
       # elif self.solver_name == 'glpk':
       #     _sparse_model_setup_glpk(self.solver_model,
       #                              fvec, amat, bvec, aeqmat, beq,
       #                              lbvec, ubvec, variable_names)
       # elif self.solver_name == 'scipy':
       #     _sparse_model_setup_scipy(self.solver_model,
       #                               fvec, amat, bvec, aeqmat, beq,
       #                               lbvec, ubvec, variable_names)


    def optimize(self):
        """
        Call the optimization routine of the underlying solver
        """
        if self.solver_name == 'gurobi':
            self.solver_model.optimize()
            self.status = self.solver_model.status
        elif self.solver_name == 'soplex':
            self.solver_model.optimize()
            self.status = self.solver_model.getStatus()


class MILPModel(LPModel):
    """
    Wrapper class for handling MILP problems of the form
     min  f'*x + fbar'*xbar
     s.t. A*x + Abar*xbar <= b
                    Aeq*x == beq
                       lb <= x <= ub
                        x in R^n
                     xbar in B^m
        and the variable names stored in the list "variable_names"
    """
    def sparse_mip_model_setup(self, fvec, barf, amat, baramat, bvec, aeqmat,
                               beq, lbvec, ubvec, variable_names):
        """
        cf. sparse_model_setup for the LPModel class
        """
        n_booles = len(barf)
        m_aeqmat = aeqmat.shape[0]
        if self.solver_name == 'gurobi':
            _sparse_model_setup_gurobi(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # A,
                bvec, # b
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # Aeq
                beq, # beq
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lb
                np.vstack([ubvec, np.ones((n_booles, 1))]), # ub
                variable_names,
                nbooles=n_booles)
        elif self.solver_name == 'soplex':
            _sparse_model_setup_soplex(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # A,
                bvec, # b
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # Aeq
                beq, # beq
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lb
                np.vstack([ubvec, np.ones((n_booles, 1))]), # ub
                variable_names,
                nbooles=n_booles)


# GUROBI - specifics ##########################################################
def _sparse_model_setup_gurobi(model, fvec, amat, bvec, aeqmat, beq, lbvec,
                               ubvec, variable_names, nbooles=0):
    """
    We set up the following (continuous) LP for gurobipy:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for binary/integer variables/ coupling constraints
        between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    model.setObjective(MINIMIZE)
    model.setParam('OutputFlag', 0) # DEBUG

    n_x = lbvec.size - nbooles
    x_variables = [model.addVar(lb=lbvec[i],
                                ub=ubvec[i],
                                obj=fvec[i],
                                name=variable_names[i]) for i in range(n_x)]

    x_variables += [model.addVar(lb=lbvec[i],
                                 ub=ubvec[i],
                                 obj=fvec[i],
                                 name=variable_names[i],
                                 vtype=gurobipy.GRB.BINARY) for i in range(n_x, n_x+nbooles)]

    _add_sparse_constraints_gurobi(model, amat, bvec, x_variables, gurobipy.GRB.LESS_EQUAL)
    _add_sparse_constraints_gurobi(model, aeqmat, beq, x_variables, gurobipy.GRB.EQUAL)
    model.update()


def _add_sparse_constraints_gurobi(model, amat, bvec, x_variables, sense):
    """
    bulk-add constraints of the form A*x <sense> b to a gurobipy model
    Note that we do not update the model!
    inspired by https://stackoverflow.com/questions/22767608/
                                  sparse-matrix-lp-problems-in-gurobi-python
    """
    nrows = bvec.size
    #A.sort_indices()
    #Aeq.sort_indices()
    for i in range(nrows):
        #MAYBE: It could be even faster to use the rows of the csr_matrix
        #directly, starting from gurobi 9.0 there is some form of matrix
        #interface
        start = amat.indptr[i]
        end = amat.indptr[i+1]
        variables = [x_variables[j] for j in amat.indices[start:end]]
        coeff = amat.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=sense, rhs=bvec[i])


# SOPLEX - specifics ##########################################################
def _sparse_model_setup_soplex(model, fvec, amat, bvec, aeqmat, beq, lbvec,
                               ubvec, variable_names, nbooles=0):
    """
    We set up the following LP for pyscipopt:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for binary/integer variables/ coupling constraints
        between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    model.hideOutput() # DEBUG
    n_x = lbvec.size - nbooles
    x_variables = [model.addVar(variable_names[i],
                                vtype='C',
                                lb=lbvec[i],
                                ub=ubvec[i],
                                obj=fvec[i]) for i in range(n_x)]
    x_variables += [model.addVar(variable_names[i],
                                 vtype='B',
                                 lb=lbvec[i],
                                 ub=ubvec[i],
                                 obj=fvec[i]) for i in range(n_x, n_x+nbooles)]
    tmp = pyscipopt.quicksum([fvec[i,0]*x_variables[i] for i in range(len(fvec))])
    model.setObjective(tmp,
                       sense=MINIMIZE) # Doppelt healt besser? Jetzt setzen wir fvec zweimal
    model.data = x_variables
    _add_sparse_constraints_soplex(model, amat, bvec, x_variables, 'LEQ')
    _add_sparse_constraints_soplex(model, aeqmat, beq, x_variables, 'EQ')


def _add_sparse_constraints_soplex(model, amat, bvec, x_variables, sense):
    """
    bulk-add constraints of the form A*x <sense> b to a soplex model
    """
    nrows = bvec.size
    #A.sort_indices()
    #Aeq.sort_indices()
    for i in range(nrows):
        start = amat.indptr[i]
        end = amat.indptr[i+1]
        variables = [x_variables[j] for j in amat.indices[start:end]]
        coeff = amat.data[start:end]
        expr = pyscipopt.quicksum([coeff[j]*variables[j] for j in range(len(coeff))])
        if sense=='LEQ':
            model.addCons(expr <= bvec[i])
        elif sense=='EQ':
            model.addCons(expr == bvec[i])


# GLPK specifics ##############################################################


# SCIPY specifics #############################################################
class _Scipy_LP_model():
    """
    Just a simple container for an LP problem to be solved with
    scipy.optimize.linprog
    """
    def __init__(self):
        pass

    def setup_sparse_problem(self, fvec, amat, bvec, aeqmat, beq, lbvec, ubvec,
                             variable_names):
        """
        Just collect the data
        """
        self.c = fvec
        self.A_ub = amat
        self.b_ub = bvec
        self.A_eq = aeqmat
        self.b_eq = beq
        self.bounds = np.hstack([lbvec, ubvec])
        self.variable_names = variable_names