"""
Wrapper to interface PyrrrateFBa models and the underlying optimal control
framework
"""
import sys
import numpy as np
import scipy.sparse as sp
from .optimization.lp import INFINITY, EPSILON, BIGM, MINUSBIGM


class Matrrrices:
    """
    Class that contains the matrices and vectors of the MILP:
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
    TODO: Create names for the constraints such that is becomes easier to "play
          with them" (meaning: to relax some or see which make the problem infeasible)
    Class Matrrrices has fields:
        phi1
        phi2
        phi3
        smat1
        smat2
        smat3
        smat4
        f_1
        lbvec
        ubvec
        hmaty
        hmatu
        hvec
        - matrix_B_y:  hbmaty
        - matrix_B_u:  hbmatu
        - matrix_B_x:  hbmatx
        - vec_B:       hbvec
        bmaty0
        bmatyend
        b_bndry
    """

    def __init__(self, model, run_rdeFBA=True):
        if run_rdeFBA and not model.is_rdeFBA:
            sys.exit('Cannot perform r-deFBA on a deFBA model!')

        self.construct_vectors(model)   # TODO: Where is f_1?
        self.construct_objective(model) # TODO: It would be great to be more flexible here
        self.construct_boundary(model)
        self.construct_reactions(model)
        self.construct_flux_bounds(model)
        self.construct_mixed(model)
        if run_rdeFBA:
            self.construct_fullmixed(model)
        # corresponding deFBA model is obtained by omitting the regulatory constraints
        else:
            # matrices must have (at least) one row to keep linprog working
            # TODO: Change this behavior in lp
            self.matrix_B_y = np.zeros((1, len(self.y_vec)), dtype=float)
            self.matrix_B_u = np.zeros((1, len(self.u_vec)), dtype=float)
            self.matrix_B_x = np.zeros((1, 1), dtype=float)
            self.vec_B = np.array([0.0])

    def construct_vectors(self, model):
        """
        construct vectors containing the IDs of dynamical species, reactions,
        and boolean variables, respectively
        """

        # species vector y contains only dynamical species, i.e.:
        y_vec = []
        # dynamical external species
        for ext in model.extracellular_dict.keys():
            if not model.extracellular_dict[ext]['constant'] and not model.extracellular_dict[ext]['boundaryCondition']:
                y_vec.append(ext)
        # and all macromolecular species
        y_vec = y_vec + list(model.macromolecules_dict.keys())

        # reactions vector u contains all reactions except for degradation reactions
        u_vec = []
        for rxn in model.reactions_dict.keys():
            if np.isfinite(model.reactions_dict[rxn]['kcatForward']):
                u_vec.append(rxn)

        # boolean variables species vector x contains all boolean variables
        # can occur multiple times in different rules/events
        x_vec = []
        # first iterate through events for boolean variables controlled by continuous variables
        for event in model.events_dict.keys():
            for variable in model.events_dict[event]['listOfAssignments']:
                if variable not in x_vec:
                    x_vec.append(variable)
        # then interate through rules for boolean variables regulating fluxes
        for rule in model.rules_dict.keys():
            if 'indicators' in model.rules_dict[rule]:
                for indicator in model.rules_dict[rule]['indicators']:
                    if indicator not in x_vec:
                        x_vec.append(indicator)
            elif 'bool_parameter' in model.rules_dict[rule]:
                parameter = model.rules_dict[rule]['bool_parameter']
                if parameter not in x_vec:
                    x_vec.append(parameter)


        self.y_vec = y_vec
        self.u_vec = u_vec
        self.x_vec = x_vec

    def construct_objective(self, model, phi2=None, phi3=None):
        """
        constructs objective vectors Phi_1, Phi_2 and Phi_3.
        """

        self.phi1 = np.zeros(len(self.y_vec), dtype=float)

        for macrom in model.macromolecules_dict.keys():
            self.phi1[self.y_vec.index(macrom)] = -model.macromolecules_dict[macrom]['objectiveWeight']

        if phi2:
            self.phi2 = phi2
        else:
            self.phi2 = np.zeros(len(self.y_vec), dtype=float)

        if phi3:
            self.phi3 = phi3
        else:
            self.phi3 = np.zeros(len(self.y_vec), dtype=float)

    def construct_boundary(self, model):
        """
        construct matrices to enforce boundary conditions
        """
        # initialize matrices
        matrix_start = np.zeros((0, len(self.y_vec)), dtype=float)
        # how to encode cyclic behaviour in SBML?
        # matrix_end = np.zeros((0, len(self.y_vec)), dtype=float)
        vec_bndry = np.zeros((0, 1), dtype=float)
        # append rows if initialAmount is given and fill bndry vector
        for ext in model.extracellular_dict.keys():
            if np.isnan(model.extracellular_dict[ext]["initialAmount"]):
                pass
            else:
                amount = float(model.extracellular_dict[ext]["initialAmount"])
                # only for dynamical extracellular species
                if ext in self.y_vec:
                    new_row = np.zeros(len(self.y_vec), dtype=float)
                    new_row[self.y_vec.index(ext)] = 1
                    matrix_start = np.append(matrix_start, [new_row], axis=0)
                    vec_bndry = np.append(vec_bndry, [[amount]], axis=0)
        for macrom in model.macromolecules_dict.keys():
            if np.isnan(model.macromolecules_dict[macrom]["initialAmount"]):
                pass
            else:
                amount = float(model.macromolecules_dict[macrom]["initialAmount"])
                new_row = np.zeros(len(self.y_vec), dtype=float)
                new_row[self.y_vec.index(macrom)] = 1
                matrix_start = np.append(matrix_start, [new_row], axis=0)
                vec_bndry = np.append(vec_bndry, [[amount]], axis=0)

        # enforce that the weighted sum of all macromolecules is 1
        # only if there is a macromolecule without an initialAmount specified
        enforce_biomass = False
        for macrom in model.macromolecules_dict.keys():
            if np.isnan(model.macromolecules_dict[macrom]['initialAmount']):
                enforce_biomass = True
                break
        if enforce_biomass:
            weights_row = np.zeros(len(self.y_vec), dtype=float)
            for macrom in model.macromolecules_dict.keys():
                weight = float(model.macromolecules_dict[macrom]["molecularWeight"])
                weights_row[self.y_vec.index(macrom)] = weight
            matrix_start = np.append(matrix_start, [weights_row], axis=0)
            vec_bndry = np.append(vec_bndry, [[1.0]], axis=0)

        self.matrix_start = sp.csr_matrix(matrix_start)
        self.matrix_end = sp.csr_matrix(np.zeros((self.matrix_start.shape), dtype=float))
        self.vec_bndry = vec_bndry

    def construct_reactions(self, model):
        """
        construct matrices S1, S2, S3, S4
        """

        # select rows of species with QSSA
        # # (first entries in stoichiometric matrix belong to extreacellular species,
        # followed by internal metabolites)
        # initialize with indices of internal metabolites
        rows_qssa = list(
            range(len(model.extracellular_dict), len(model.extracellular_dict) + len(model.metabolites_dict)))
        # add non-dynamical extracellualar species
        for row_index, ext in enumerate(model.extracellular_dict):
            if model.extracellular_dict[ext]['constant'] or model.extracellular_dict[ext]['boundaryCondition']:
                rows_qssa.append(row_index)

        # select columns of degradation reactions
        cols_deg = []
        for col_index, rxn in enumerate(model.reactions_dict):
            if np.isnan(model.reactions_dict[rxn]['kcatForward']):
                cols_deg.append(col_index)

        # S1: QSSA species
        smat1 = model.stoich[rows_qssa, :]
        smat1 = np.delete(smat1, cols_deg, 1)

        # S2: dynamical species
        smat2 = np.delete(model.stoich, rows_qssa, 0)
        smat2 = np.delete(smat2, cols_deg, 1)

        # S3: QSSA species
        smat3 = model.stoich_degradation[rows_qssa, :]
        smat3 = np.delete(smat3, rows_qssa, 1)  # here: rows_qssa = cols_qssa

        # S4: dynamical species
        smat4 = np.delete(model.stoich_degradation, rows_qssa, 0)
        smat4 = np.delete(smat4, rows_qssa, 1)  # here: rows_qssa = cols_qssa

        self.smat1 = sp.csr_matrix(smat1)
        self.smat2 = sp.csr_matrix(smat2)
        self.smat3 = sp.csr_matrix(smat3)
        self.smat4 = sp.csr_matrix(smat4)

    def construct_flux_bounds(self, model):
        """
        construct vectors lb, ub
        """
        lbvec = np.array(len(self.u_vec) * [[0.0]])
        ubvec = np.array(len(self.u_vec) * [[INFINITY]])

        # flux bounds determined by regulation are not considered here
        for index, rxn in enumerate(model.reactions_dict):
            # lower bounds
            try:
                lbvec[index] = float(model.reactions_dict[rxn]['lowerFluxBound'])
            except KeyError:
                pass
            # upper bounds
            try:
                ubvec[index] = float(model.reactions_dict[rxn]['upperFluxBound'])
            except KeyError:
                pass

        self.lbvec = lbvec
        self.ubvec = ubvec

    def construct_fullmixed(self, model):
        """
        construct matrices for regulation
        """

        epsilon = EPSILON
        u = BIGM
        l = MINUSBIGM

        # control of discrete jumps

        # initialize matrices
        n_assignments = sum([len(evnt['listOfAssignments']) 
                             for evnt in model.events_dict.values()])
        matrix_B_y_1 = np.zeros((n_assignments, len(self.y_vec)), dtype=float)
        matrix_B_u_1 = np.zeros((n_assignments, len(self.u_vec)), dtype=float)
        matrix_B_x_1 = np.zeros((n_assignments, len(self.x_vec)), dtype=float)
        vec_B_1 = np.array(n_assignments * [[0.0]])

        event_index = 0
        for event in model.events_dict.keys():
            variables = model.events_dict[event]['variable'].split(' + ')
            # Difference between geq and gt??
            if model.events_dict[event]['relation'] == 'geq' or model.events_dict[event]['relation'] == 'gt':
                for i, affected_bool in enumerate(model.events_dict[event]['listOfAssignments']):
                    for variable in variables:
                        # boolean variable depends on species amount
                        if variable in self.y_vec:
                            species_index = self.y_vec.index(variable)
                            matrix_B_y_1[event_index][species_index] = 1
                            if model.events_dict[event]['listOfEffects'][i] == 0:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = epsilon + u
                                vec_B_1[event_index] = model.events_dict[event]['threshold'] + u
                            elif model.events_dict[event]['listOfEffects'][i] == 1:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = - (epsilon + u)
                                vec_B_1[event_index] = model.events_dict[event]['threshold'] - epsilon

                        # boolean variable depends on flux
                        elif variable in self.u_vec:
                            flux_index = self.u_vec.index(variable)
                            matrix_B_u_1[event_index][flux_index] = 1
                            if model.events_dict[event]['listOfEffects'][i] == 0:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = epsilon + u
                                vec_B_1[event_index] = model.events_dict[event]['threshold'] + u
                            elif model.events_dict[event]['listOfEffects'][i] == 1:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = - (epsilon + u)
                                vec_B_1[event_index] = model.events_dict[event]['threshold'] - epsilon
                        else:
                            print(variable + ' not defined as Species or Reaction!')
                event_index += 1


            # TODO Difference between leq and lt??
            elif model.events_dict[event]['relation'] == 'leq' or model.events_dict[event]['relation'] == 'lt':
                for i, affected_bool in enumerate(model.events_dict[event]['listOfAssignments']):
                    for variable in variables:
                        # boolean variable depends on species amount
                        if variable in self.y_vec:
                            species_index = self.y_vec.index(variable)
                            matrix_B_y_1[event_index][species_index] = -1
                            if model.events_dict[event]['listOfEffects'][i] == 0:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = - l
                                vec_B_1[event_index] = -model.events_dict[event]['threshold'] - l
                            elif model.events_dict[event]['listOfEffects'][i] == 1:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = l
                                vec_B_1[event_index] = -model.events_dict[event]['threshold']

                        # boolean variable depends on flux
                        elif variable in self.u_vec:
                            flux_index = self.u_vec.index(variable)
                            matrix_B_u_1[event_index][flux_index] = -1
                            if model.events_dict[event]['listOfEffects'][i] == 0:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = - l
                                vec_B_1[event_index] = -model.events_dict[event]['threshold'] - l
                            elif model.events_dict[event]['listOfEffects'][i] == 1:
                                matrix_B_x_1[event_index][self.x_vec.index(affected_bool)] = l
                                vec_B_1[event_index] = -model.events_dict[event]['threshold']
                        else:
                            print(variable + ' not defined as Species or Reaction!')
                event_index += 1



        n_rules = 0
        for rule in model.rules_dict.keys():
            if 'reactionID' in model.rules_dict[rule]:
                n_rules += 1
            elif 'operator' in model.rules_dict[rule]:
                n_rules += len(model.rules_dict[rule]['indicators']) + 1

        matrix_B_y_2 = np.zeros((n_rules, len(self.y_vec)), dtype=float)
        matrix_B_u_2 = np.zeros((n_rules, len(self.u_vec)), dtype=float)
        matrix_B_x_2 = np.zeros((n_rules, len(self.x_vec)), dtype=float)
        vec_B_2 = np.array(n_rules * [[0.0]])

        rule_row_index = 0

        for rule in model.rules_dict.keys():
            # Control of continuous dynamics by discrete states
            if 'reactionID' in model.rules_dict[rule]:
                rxn_index = self.u_vec.index(model.rules_dict[rule]['reactionID'])
                par_index = self.x_vec.index(model.rules_dict[rule]['bool_parameter'])
                if model.rules_dict[rule]['direction'] == 'lower':
                    matrix_B_u_2[rule_row_index][rxn_index] = -1
                    if np.isnan(model.rules_dict[rule]['threshold']):
                        matrix_B_x_2[rule_row_index][par_index] = l
                    else:
                        matrix_B_x_2[rule_row_index][par_index] = float(model.rules_dict[rule]['threshold'])
                if model.rules_dict[rule]['direction'] == 'upper':
                    matrix_B_u_2[rule_row_index][rxn_index] = 1
                    if np.isnan(model.rules_dict[rule]['threshold']):
                        matrix_B_x_2[rule_row_index][par_index] = -u
                    else:
                        matrix_B_x_2[rule_row_index][par_index] = float(model.rules_dict[rule]['threshold'])
                rule_row_index += 1 # only requires one line, i.e. one inequality
            # Boolean Algebra Rules
            elif 'operator' in model.rules_dict[rule]:
                variable_index = self.x_vec.index(rule)
                # Discriminate AND and OR
                if model.rules_dict[rule]['operator'] == 304: # AND
                    # first inequality (the one containing all variables)
                    matrix_B_x_2[rule_row_index][variable_index] = -1
                    vec_B_2[rule_row_index] = len(model.rules_dict[rule]['indicators']) - 1
                    for i in range(len(model.rules_dict[rule]['indicators'])):
                        indicator_index = self.x_vec.index(model.rules_dict[rule]['indicators'][i])
                        # positive in the first inequality containing all variables
                        matrix_B_x_2[rule_row_index][indicator_index] = 1
                        # negative in another (dependent variable is positive there)
                        matrix_B_x_2[rule_row_index + i + 1][indicator_index] = -1
                        matrix_B_x_2[rule_row_index + i + 1][variable_index] = 1
                    rule_row_index += len(model.rules_dict[rule]['indicators']) + 1
                elif model.rules_dict[rule]['operator'] == 306: # OR
                    # first inequality (the one containing all variables)
                    matrix_B_x_2[rule_row_index][variable_index] = 1
                    for i in range(len(model.rules_dict[rule]['indicators'])):
                        indicator_index = self.x_vec.index(model.rules_dict[rule]['indicators'][i])
                        # negative in the first inequality containing all variables
                        matrix_B_x_2[rule_row_index][indicator_index] = -1
                        # positive in another (dependent variable is positive there)
                        matrix_B_x_2[rule_row_index + i + 1][indicator_index] = 1
                        matrix_B_x_2[rule_row_index + i + 1][variable_index] = -1
                    rule_row_index += len(model.rules_dict[rule]['indicators']) + 1


        self.matrix_B_y = sp.csr_matrix(np.vstack((matrix_B_y_1, matrix_B_y_2)))
        self.matrix_B_u = sp.csr_matrix(np.vstack((matrix_B_u_1, matrix_B_u_2)))
        self.matrix_B_x = sp.csr_matrix(np.vstack((matrix_B_x_1, matrix_B_x_2)))
        self.vec_B = np.vstack((vec_B_1, vec_B_2))

    def construct_mixed(self, model):
        """
        constructs matrices H_y and H_u
        """
        # TODO: What if more than one enzyme catalyzes one reactions? One should at least check.
        # check whether the model contains quota compounds
        n_quota = 0  # number of quota and storage compounds
        for macrom in model.macromolecules_dict.keys():
            if model.macromolecules_dict[macrom]['speciesType'] == 'quota':
                n_quota += 1

        if n_quota > 0:
            matrix_y_1 = self.construct_Hb(model, n_quota)
            matrix_u_1 = np.zeros((matrix_y_1.shape[0], len(self.u_vec)), dtype=float)

        matrix_y_2, matrix_u_2 = self.construct_HcHe(model)

        # check whether the model contains a maintenance reaction; if so, pass index
        main_rxn = None
        for rxn in model.reactions_dict.keys():
            if model.reactions_dict[rxn]['maintenanceScaling'] > 0:
                main_rxn = rxn
                matrix_y_3, matrix_u_3 = self.construct_Hm(model, main_rxn)
                break

        # stacking of the resulting matrices
        if n_quota > 0:
            if main_rxn:
                self.matrix_u = sp.csr_matrix(np.vstack((matrix_u_1, matrix_u_2, matrix_u_3)))
                self.matrix_y = sp.csr_matrix(np.vstack((matrix_y_1, matrix_y_2, matrix_y_3)))
            else:
                self.matrix_u = sp.csr_matrix(np.vstack((matrix_u_1, matrix_u_2)))
                self.matrix_y = sp.csr_matrix(np.vstack((matrix_y_1, matrix_y_2)))
        elif main_rxn:
            self.matrix_u = sp.csr_matrix(np.vstack((matrix_u_2, matrix_u_3)))
            self.matrix_y = sp.csr_matrix(np.vstack((matrix_y_2, matrix_y_3)))
        else:
            self.matrix_u = sp.csr_matrix(matrix_u_2)
            self.matrix_y = sp.csr_matrix(matrix_y_2)
        # h always contains only zeros
        self.vec_h = np.zeros((self.matrix_u.shape[0], 1), dtype=float)

    def construct_Hb(self, model, n_rows):
        """
        Construct the H_B matrix
        """

        HB_matrix = np.zeros((n_rows, len(self.y_vec)))
        i = 0  # row (quota) counter
        if i < n_rows:  # stop iterating when all quota compounds have been considered
            for quota in model.macromolecules_dict.keys():
                if model.macromolecules_dict[quota]['speciesType'] == 'quota':
                    for macrom in model.macromolecules_dict.keys():
                        if macrom == quota:
                            HB_matrix[i][self.y_vec.index(macrom)] = (model.macromolecules_dict[quota][
                                                                          'biomassPercentage'] - 1) * \
                                                                     model.macromolecules_dict[macrom][
                                                                         'molecularWeight']
                        else:
                            HB_matrix[i][self.y_vec.index(macrom)] = model.macromolecules_dict[quota][
                                                                         'biomassPercentage'] * \
                                                                     model.macromolecules_dict[macrom][
                                                                         'molecularWeight']
                    i += 1

        return HB_matrix

    def construct_HcHe(self, model):
        """
        Construct matrices for enzyme capacity constraints: H_C and filter matrix H_E
        """
        # calculate number of rows in H_C and H_E:
        n_rev = []  # list containing number of reversible rxns per enzyme

        # iterate over enzymes
        for enzyme in model.macromolecules_dict.keys():
            e_rev = 0  # number of reversible reactions catalyzed by this enzyme
            enzyme_catalyzed_anything = False
            # iterate over reactions
            for rxn in model.reactions_dict.keys():
                if model.reactions_dict[rxn]['geneProduct'] == enzyme:
                    enzyme_catalyzed_anything = True
                    if model.reactions_dict[rxn]['reversible']:
                        e_rev = e_rev + 1
            if enzyme_catalyzed_anything:
                n_rev.append(e_rev)

        # initialize matrices
        n_rows = sum(2 ** i for i in n_rev)  # number of rows in H_C and H_E
        HC_matrix = np.zeros((n_rows, len(self.u_vec)), dtype=float)
        HE_matrix = np.zeros((n_rows, len(self.y_vec)), dtype=float)

        # fill matrices
        e = 0  # enzyme counter
        i = 0  # row counter

        # iterate over enzymes
        for enzyme in model.macromolecules_dict.keys():
            # if macromolecule doesn't catalyze any reaction (e.g. transcription factors), it won't be regarded
            enzyme_catalyzes_anything = False
            # n_rev contains a number for each catalytically active enzyme
            if e < len(n_rev):
                # increment macromolecule counter
                c_rev = 0  # reversible-reaction-per-enzyme counter
                # iterate over reactions
                if c_rev <= n_rev[e]:
                    for rxn in model.reactions_dict.keys():
                        # if there is a reaction catalyzed by this macromolecule (i.e. it is a true enzyme)
                        if model.reactions_dict[rxn]['geneProduct'] == enzyme:
                            enzyme_catalyzes_anything = True
                            # reversible reactions
                            if model.reactions_dict[rxn]['reversible']:
                                # boolean variable specifies whether to include forward or backward k_cat
                                fwd = True
                                # in order to cover all possible combinations of reaction fluxes
                                for r in range(2 ** n_rev[e]):
                                    if fwd:
                                        HC_matrix[i + r][self.u_vec.index(rxn)] = np.reciprocal(
                                            model.reactions_dict[rxn]['kcatForward'])
                                        HE_matrix[i + r][self.y_vec.index(enzyme)] = 1
                                        r = r + 1
                                        # true after half of the combinations for the first reversible reaction
                                        # true after 1/4 of the combinations for the second reversible reaction
                                        # and so on.
                                        if np.mod(r, 2 ** n_rev[e] / 2 ** (c_rev + 1)) == 0:
                                            fwd = False
                                    else:
                                        HC_matrix[i + r][self.u_vec.index(rxn)] = -1 * np.reciprocal(
                                            model.reactions_dict[rxn]['kcatBackward'])
                                        HE_matrix[i + r][self.y_vec.index(enzyme)] = -1
                                        r = r + 1
                                        # as above, fwd will be switched after 1/2, 1/4, ... of the possible combinations
                                        if np.mod(r, 2 ** n_rev[e] / 2 ** (c_rev + 1)) == 0:
                                            fwd = True
                                c_rev = c_rev + 1
                            # irreversible reactions
                            else:
                                # simply enter 1/k_cat for each combination
                                # (2^0 = 1 in case of an enzyme that only catalyzes irreversible reactions)
                                for r in range(2 ** n_rev[e]):
                                    HC_matrix[i + r][self.u_vec.index(rxn)] = np.reciprocal(
                                        model.reactions_dict[rxn]['kcatForward'])
                                    HE_matrix[i + r][self.y_vec.index(enzyme)] = 1
            if enzyme_catalyzes_anything:
                i = i + 2 ** n_rev[e]
                e = e + 1

        return -HE_matrix, HC_matrix

    def construct_Hm(self, model, main_rxn):
        """
        Constructs the H_M matrix (assumption: there is at most one maintenance reaction)
        """

        main_scaling = model.reactions_dict[main_rxn]['maintenanceScaling']
        main_index = self.u_vec.index(main_rxn)

        # matrix has entry -1 where the column corresponds to the maintenance reaction, zeros elsewhere
        matrix_HM_u = np.zeros(len(self.u_vec), dtype=float)
        matrix_HM_u[main_index] = -1.0

        # entries in matrix correspond to weights * maintenanceScaling
        matrix_HM_y = np.zeros(len(self.y_vec), dtype=float)
        for macrom in model.macromolecules_dict.keys():
            matrix_HM_y[self.y_vec.index(macrom)] = main_scaling * model.macromolecules_dict[macrom]['molecularWeight']

        return matrix_HM_y, matrix_HM_u
