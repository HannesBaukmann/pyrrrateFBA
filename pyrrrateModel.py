"""
Main class for PyrrrateFBA models
"""

import numpy as np
from .simulation import fba
#from .simulation.fba import perform_fba

class Model():
    """
    PyrrrateFBA Models
    """
    def __init__(self, ram_model):
        self.name = ram_model.name
        self.is_rdeFBA = ram_model.is_rdeFBA
        self.extracellular_dict = ram_model.extracellular_dict
        self.metabolites_dict = ram_model.metabolites_dict
        self.macromolecules_dict = ram_model.macromolecules_dict
        self.reactions_dict = ram_model.reactions_dict
        self.qualitative_species_dict = ram_model.qualitative_species_dict
        self.events_dict = ram_model.events_dict
        self.rules_dict = ram_model.rules_dict
        self.stoich = ram_model.stoich
        self.stoich_degradation = ram_model.stoich_degradation


    def print_numbers(self):
        """
        display relevant integer values of a pyrrrateFBA model
        """
        quota = 0
        stor = 0

        spon = 0
        main = 0

        for mm in self.macromolecules_dict.keys():
            if self.macromolecules_dict[mm]['speciesType'] == 'quota':
                quota += 1
            elif self.macromolecules_dict[mm]['speciesType'] == 'storage':
                stor += 1

        for rxn in self.reactions_dict.keys():
            if self.reactions_dict[rxn]['maintenanceScaling'] > 0.0:
                main += 1
            if not self.reactions_dict[rxn]['geneProduct']:
                spon += 1

        print('species\t\t\t\t' + str(len(self.extracellular_dict) + len(self.metabolites_dict) \
              + len(self.macromolecules_dict)) \
              + '\n\t metabolites\t\t' + str(len(self.extracellular_dict) \
              + len(self.metabolites_dict)) \
              + '\n\t\t extracellular\t' + str(len(self.extracellular_dict)) \
              + '\n\t\t intracellular\t' + str(len(self.metabolites_dict)) \
              + '\n\t macromolecules\t\t' + str(len(self.macromolecules_dict)) \
              + '\n\t\t enzymes\t' + str(len(self.macromolecules_dict) - quota - stor) \
              + '\n\t\t quota\t\t' + str(quota) \
              + '\n\t\t storage\t' + str(stor) \
              + '\n reactions\t\t\t' + str(len(self.reactions_dict)) \
              + '\n\t uptake\t\t' \
              + '\n\t metabolic\t\t' \
              + '\n\t translation\t\t' \
              + '\n\t degradation\t\t' + str(np.count_nonzero(self.stoich_degradation)) \
              + '\n\t spontaneous\t\t' + str(spon) \
              + '\n\t maintenance\t\t' + str(main) \
              + '\n regulation\t\t\t\t' \
              + '\n\t rules\t\t' + str(len(self.events_dict)/2) \
              + '\n\t regulatory proteins\t\t' + str(len(self.qualitative_species_dict)) \
              + '\n\t regulated reactions\t\t')

    def fba(self, objective=None, maximize=True):
        """
        performs Flux Balance Analysis
        """
        sol = fba.perform_fba(self, objective=None, maximize=True)
        return sol
    
    
    #def rdeFBA(self, tspan, varphi, run_rdeFBA=True):
    def rdeFBA(self, tspan, varphi, **kwargs):
        """
        Perform (r)deFBA
        """
        kwargs['t_0'] = tspan[0]
        kwargs['t_end'] = tspan[-1]
        kwargs['varphi'] = varphi
        #sol = fba.perform_rdefba(self, run_rdeFBA=run_rdeFBA, varphi=varphi, t_0=tspan[0], t_end=tspan[-1])
        sol = fba.perform_rdefba(self, **kwargs)

        return sol
        
"""
        import gurobipy as gp
        from gurobipy import GRB

        brxns = []

        if not objective:
            # default objective is (first) biomass function
            for rxn in self.reactions_dict.keys():
                if 'biomass' in rxn:
                    brxns = [list(self.reactions_dict.keys()).index(rxn)]
            if not brxns:
                print('Could not find biomass reaction.' \
                      + ' Please specify reaction flux to be optimized.')
                return None
        else:
            # check whether rxn exists
            brxns = [list(self.reactions_dict.keys()).index(objective)]

        S = self.stoich
        rows, cols = S.shape

        lb = [None] * cols  # lower bounds on v
        ub = [None] * cols  # upper bounds on v

        for index, rxn in enumerate(self.reactions_dict.keys(), start=0):
            try:
                ub[index] = self.reactions_dict[rxn]['upperFluxBound']
            except KeyError:
                ub[index] = 1000
            if self.reactions_dict[rxn]['reversible']:
                try:
                    lb[index] = self.reactions_dict[rxn]['lowerFluxBound']
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
        model_vars = []
        for j in range(cols):
            model_vars.append(gpmodel.addVar(lb=lb[j], ub=ub[j], vtype=GRB.CONTINUOUS))

        # Populate S matrix
        for i in range(rows):
            expr = gp.LinExpr()
            for j in range(cols):
                if S[i, j] != 0:
                    expr += S[i, j] * model_vars[j]
            gpmodel.addConstr(expr, GRB.EQUAL, 0)

        # Populate objective
        obj = gp.LinExpr()
        for j in range(cols):
            if c[j] != 0:
                obj += c[j] * model_vars[j]
        gpmodel.setObjective(obj, GRB.MAXIMIZE)

        # Solve
        gpmodel.optimize()

        # Save optimal flux for biomass reactions
        if gpmodel.status == GRB.Status.OPTIMAL:
            x = gpmodel.getAttr('x', model_vars)
            sols = [x[i] for i in brxns]

        return sols
"""