"""
Collection of routines for Linear Programming
"""

import gurobipy # TODO Make this dependent on installed/configured solvers

# Module constants
INFINITY = gurobipy.GRB.INFINITY
OPTIMAL = gurobipy.GRB.OPTIMAL


class LPModel(object):
    """
    Simple wrapper class to handle various LP solvers in a unified way
    TODO - add more solvers, more options, ILPs, and MILPs etc. pp.
    """
    def __init__(self, name=""):
        self.name = name
        self.solver_name = "gurobi" # TODO: More options/more clever handling
        self.solver_model = gurobipy.Model()
        self.status = "Unknown"


    def get_solution(self):
        """
        Output the solution vector of the LP if already calculated
        """
        if self.status != OPTIMAL:
            return None
        return self.solver_model.x


    def sparse_model_setup(self, f, A, b, Aeq, beq, lb, ub, variable_names):
        """
        Fill the "solver_model" of the underlying LP solver class with the
        following LP:
            min f'*x
            s.t. A*x <= b
                 Aeq*x == beq
                 lb <= x <= ub
            and the variable names stored in the list "variable_names"
        """
        _sparse_model_setup_gurobi(self.solver_model,
                                   f, A, b, Aeq, beq,
                                   lb, ub, variable_names)


    def optimize(self):
        """
        Call the optimization routine of the underlying solver
        """
        self.solver_model.optimize()
        self.status = self.solver_model.status




def _sparse_model_setup_gurobi(model, f, A, b, Aeq, beq, lb, ub, variable_names):
    """
    We set up the following LP for gurobipy:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances
    inspired by https://stackoverflow.com/questions/22767608/
                                  sparse-matrix-lp-problems-in-gurobi-python

    TODO:
     - Include possibility for binary/integer variables/ coupling constraints
        between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices?

    # TIP MAYBE: We could build a class around this/use an LP-class around this
    """
    model.setObjective(gurobipy.GRB.MINIMIZE)

    nx = lb.size
    x_variables = [model.addVar(lb=lb[i],
                                ub=ub[i],
                                obj=f[i],
                                name=variable_names[i]) for i in range(nx)]
    nrows = b.size
    #A.sort_indices()
    #Aeq.sort_indices()
    for i in range(nrows):
        #MAYBE: It could be even faster to use the rows of the csr_matrix
        #directly, starting from gurobi 9.0 there is some form of matrix
        #interface
        start = A.indptr[i]
        end = A.indptr[i+1]
        variables = [x_variables[j] for j in A.indices[start:end]]
        coeff = A.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=b[i])

    nrows = beq.size
    for i in range(nrows):
        start = Aeq.indptr[i]
        end = Aeq.indptr[i+1]
        variables = [x_variables[j] for j in Aeq.indices[start:end]]
        coeff = Aeq.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=beq[i])

    model.update()
#    return model
