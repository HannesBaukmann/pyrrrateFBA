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
        # MAYBE: Add other fields/put the direct SBML-relations into a sub-structure


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
        sol = fba.perform_fba(self, objective=objective, maximize=maximize)
        return sol


    def rdeFBA(self, tspan, varphi, do_soa=False, **kwargs):
        """
        Perform (r)deFBA
        """
        kwargs['t_0'] = tspan[0]     # QUESTION: Is this bad practice to alter kwargs within method?
        kwargs['t_end'] = tspan[-1]
        kwargs['varphi'] = varphi
        if do_soa:
            sol = fba.perform_soa_rdeFBA(self, **kwargs)
        else:
            sol = fba.perform_rdefba(self, **kwargs)

        return sol

    # TODO output functions, especially solutions and constraint fulfillment, objective
