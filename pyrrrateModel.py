class Model(object):

    def __init__(self):

        # Create an empty variable to save the latest results of the simulation methods
        self.results = None

        if self.is_deFBA:
            import matrrrices

            self.HC_matrix = None  # Enzyme Capacity Constraint matrix
            self.HE_matrix = None  # Filter matrix for ECC matrix
            self.HM_matrix = None  # Maintenance matrix
            self.HB_matrix = None  # Biomass composition constraints

            construct_HcHe(self)
            construct_Hm(self)
            construct_Hb(self)

    def print_numbers(model):
        extra = 0
        quota = 0
        stor = 0

        spon = 0
        main = 0

        for met in model.metabolites_dict.keys():
            if model.metabolites_dict[met]['speciesType'] == 'extracellular':
                extra += 1

        for mm in model.macromolecules_dict.keys():
            if model.macromolecules_dict[mm]['speciesType'] == 'quota':
                quota += 1
            elif model.macromolecules_dict[mm]['speciesType'] == 'storage':
                stor += 1

        for rxn in model.reactions_dict.keys():
            if model.reactions_dict[rxn]['maintenanceScaling'] > 0.0:
                main += 1
            if not model.reactions_dict[rxn]['geneProduct']:
                spon += 1

        print('species\t\t\t\t' + str(len(model.metabolites_dict) + len(model.macromolecules_dict)) \
              + '\n\t metabolites\t\t' + str(len(model.metabolites_dict)) \
              + '\n\t\t extracellular\t' + str(extra) \
              + '\n\t\t intracellular\t' + str(len(model.metabolites_dict) - extra) \
              + '\n\t macromolecules\t\t' + str(len(model.macromolecules_dict)) \
              + '\n\t\t enzymes\t' + str(len(model.macromolecules_dict) - quota - stor) \
              + '\n\t\t quota\t\t' + str(quota) \
              + '\n\t\t storage\t' + str(stor) \
              + '\n reactions\t\t\t' + str(len(model.reactions_dict)) \
              + '\n\t uptake\t\t' \
              + '\n\t metabolic\t\t' \
              + '\n\t translation\t\t' \
              + '\n\t spontaneous\t\t' + str(spon) \
              + '\n\t maintenance\t\t' + str(main) \
              + '\n regulation\t\t\t\t' \
              + '\n\t rules\t\t' \
              + '\n\t regulatory proteins\t\t' \
              + '\n\t regulated reactions\t\t')

    def fba(model, objective=None, maximize=True):
        """
        performs Flux Balance Analysis
        """

        import numpy as np
        import gurobipy as gp
        from gurobipy import GRB

        brxns = []

        if not objective:
            # default objective is (first) biomass function
            for rxn in model.reactions_dict.keys():
                if 'biomass' in rxn:
                    brxns = [list(model.reactions_dict.keys()).index(rxn)]
            if not brxns:
                print('Could not find biomass reaction. Please specify reaction flux to be optimized.')
                return None
        else:
            # check whether rxn exists
            brxns = [list(model.reactions_dict.keys()).index(objective)]

        S = model.stoich
        rows, cols = S.shape

        lb = [None] * cols  # lower bounds on v
        ub = [None] * cols  # upper bounds on v

        for index, rxn in enumerate(model.reactions_dict.keys(), start=0):
            try:
                ub[index] = model.reactions_dict[rxn]['upperFluxBound']
            except KeyError:
                ub[index] = 1000
            if model.reactions_dict[rxn]['reversible']:
                try:
                    lb[index] = model.reactions_dict[rxn]['lowerFluxBound']
                except KeyError:
                    lb[index] = -1000
            else:
                lb[index] = 0

        # c vector objective function
        c = np.zeros(cols)
        if maximize:
            c[brxns] = 1
        else:
            c[brxns] = -1

        # save solutions
        sols = np.zeros(len(brxns))

        gpmodel = gp.Model()
        gpmodel.setParam('OutputFlag', False)

        # Add variables to model
        vars = []
        for j in range(cols):
            vars.append(gpmodel.addVar(lb=lb[j], ub=ub[j], vtype=GRB.CONTINUOUS))

        # Populate S matrix
        for i in range(rows):
            expr = gp.LinExpr()
            for j in range(cols):
                if S[i, j] != 0:
                    expr += S[i, j] * vars[j]
            gpmodel.addConstr(expr, GRB.EQUAL, 0)

        # Populate objective
        obj = gp.LinExpr()
        for j in range(cols):
            if c[j] != 0:
                obj += c[j] * vars[j]
        gpmodel.setObjective(obj, GRB.MAXIMIZE)

        # Solve
        gpmodel.optimize()

        # Save optimal flux for biomass reactions
        if gpmodel.status == GRB.Status.OPTIMAL:
            x = gpmodel.getAttr('x', vars)
            sols = [x[i] for i in brxns]

        return sols