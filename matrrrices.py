"""
Wrapper to interface PyrrrateFBA models and the underlying optimal control
framework
    min int_{t_0}^{t_end} e^(-varphi *t) (phi1^T y + phi1u^T u) dt + phi2^T y_0 + phi3^T y_end
     s.t.                             y' == smat2*u + smat4*y + f_2
                                       0 == smat1*u + smat3*y + f_1
                                   lbvec <= u <= ubvec
                       hmaty*y + hmatu*u <= hvec
                                       0 <= y
          hbmaty*y + hbmatu*u + hbmatx*x <= hbvec
              bmaty0*y_0 + bmatyend*y_end
            + bmatu0*u_0 + bmatuend*u_end == b_bndry
                                       y in R^{n_y}, u in R^{n_u},
                                       x in B^{n_x}

        - y_vec : IDs
        - u_vec : of the
        - x_vec : respective entries in the model
        - phi1 : phi1
        - phi2 : phi2
        - phi3 : phi3
        - smat1 : smat1
        - smat2 : smat2
        - smat3 : smat3
        - smat4: smat4

        - lbvec        : lbvec
        - ubvec        : ubvec
        - matrix_u : hmaty
        - matrix_y : hmatu
        - vec_h : hvec
        - matrix_B_y:  hbmaty
        - matrix_B_u:  hbmatu
        - matrix_B_x:  hbmatx
        - vec_B:       hbvec
        - matrix_start : bmaty0
        - matrix_end: bmatyend
        - vec_bndry : b_bndry
        #
        # NEW:
        - mixed_names: list of strings specifying the meaning of the mixed constraints
"""
import sys
import numpy as np
import scipy.sparse as sp
from .optimization.lp import INFINITY, EPSILON, BIGM, MINUSBIGM
from .util.linalg import solve_if_unique, shape_of_callable


# The order is important
MAT_FIELDS = ['y_vec', 'u_vec', 'x_vec',
              'phi1', 'phi1u', 'phi2', 'phi3',
              'smat1', 'smat2', 'smat3', 'smat4', 'f_1', 'f_2',
              'lbvec', 'ubvec',
              'matrix_u', 'matrix_y', 'vec_h', 'mixed_names',
              'matrix_B_u', 'matrix_B_y', 'matrix_B_x', 'vec_B',
              'matrix_start', 'matrix_end', 'vec_bndry', 'matrix_u_start', 'matrix_u_end']



class Matrrrices:
    """
    Class that contains the matrices and vectors of the Linear OC problem:
        Class Matrrrices has fields according to MAT_FIELDS

    TODO: - Create names for the constraints such that is becomes easier to "play
          with them" (meaning: to relax some or see which make the problem infeasible)
    TODO: - setter for some parts (e.g. initial/boundary values)
    """

    def __init__(self, model, run_rdeFBA=True, **kwargs):
        #
        if model is None:
            # We just initialize an empty model. This is just quick and dirty at the moment...
            for kw_name, kw_val in kwargs.items():
                if kw_name in MAT_FIELDS:
                    self.__dict__[kw_name] = kw_val
                else:
                    print('Could not set input ', kw_name, ' in Matrrrices construction.')
        #
        else:
            if run_rdeFBA and not model.is_rdeFBA:
                sys.exit('Cannot perform r-deFBA on a deFBA model!')

            self.construct_vectors(model)
            self.construct_objective(model)
            self.construct_boundary(model)
            self.construct_reactions(model)
            self.construct_flux_bounds(model)
            self.construct_mixed(model)
            if run_rdeFBA:
                self.construct_fullmixed(model)
            # corresponding deFBA model is obtained by omitting the regulatory constraints
            else:
                self.matrix_B_y = np.zeros((0, len(self.y_vec)), dtype=float)
                self.matrix_B_u = np.zeros((0, len(self.u_vec)), dtype=float)
                self.matrix_B_x = np.zeros((0, 0), dtype=float)
                self.vec_B = np.zeros((0, 1))
        #
        # set so-far unset fields
        self._set_unset_fields()
        self.check_dimensions()


    def extract_initial_values(self):
        """
        Boundary values: Is it possible to get a complete description
        of the initial values from the given boundary matrices?
        If so, return it
        """
        # Any constraints on the end points?
        if self.matrix_end.nnz > 0:
            print('Unable to extract initial values from model matrrrices.',
                  'End point constraints given')
            # TODO: control prints/warnings via verbosity level parameter?
            return None # MAYBE: This is technically not correct: We can have a full
            # description of initial values PLUS constraints on the end points
        y0 = solve_if_unique(self.matrix_start, self.vec_bndry)
        if y0 is None:
            print('Unable to extract initial values from model matrrrices.',
                  'System appears to be not uniquely solvable.')
            return None
        return np.expand_dims(y0, axis=1).transpose() # MAYBE: Transpose is :-(, change structure


    def check_dimensions(self):
        """
        Provide warnings/errors if dimensions are inconsistent
        Quick-and-dirty code ...
        # TODO: Check dimensions of objective vectors, check types also
        """
        n_y = len(self.y_vec)
        n_u = len(self.u_vec)
        n_x = len(self.x_vec)
        #
        n_qssa = shape_of_callable(self.smat1)[0]
        n_dyn = shape_of_callable(self.smat2)[0]
        n_mix = shape_of_callable(self.matrix_u)[0]
        n_bmix = shape_of_callable(self.matrix_B_u)[0]
        n_bndry = shape_of_callable(self.matrix_start)[0]
        #
        if shape_of_callable(self.smat1) != (n_qssa, n_u):
            raise ValueError('smat1 has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.smat1.shape[0], self.smat1.shape[1], n_qssa, n_u))
        if shape_of_callable(self.smat3) != (n_qssa, n_y):
            raise ValueError('smat3 has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.smat3.shape[0], self.smat3.shape[1], n_qssa, n_y))
        if shape_of_callable(self.f_1) != (n_qssa, 1):
            raise ValueError('f_1 has wrong dimension')
        #
        if shape_of_callable(self.smat2) != (n_dyn, n_u):
            raise ValueError('smat2 has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.smat1.shape[0], self.smat1.shape[1], n_dyn, n_u))
        if shape_of_callable(self.smat4) != (n_dyn, n_y):
            raise ValueError('smat4 has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.smat4.shape[0], self.smat4.shape[1], n_dyn, n_y))
        if shape_of_callable(self.f_2) != (n_dyn, 1):
            raise ValueError('f_2 has wrong dimension')
        #
        if shape_of_callable(self.matrix_u) != (n_mix, n_u):
            raise ValueError('matrix_u has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_u.shape[0], self.matrix_u.shape[1], n_mix, n_u))
        if shape_of_callable(self.matrix_y) != (n_mix, n_y):
            raise ValueError('matrix_y has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_y.shape[0], self.matrix_y.shape[1], n_mix, n_y))
        if shape_of_callable(self.vec_h) != (n_mix, 1):
            raise ValueError('vec_h has wrong dimension')
        if len(self.mixed_names) != n_mix:
            raise ValueError('mixed_names has wrong number of elements')
        #
        if shape_of_callable(self.matrix_B_u) != (n_bmix, n_u):
            raise ValueError('matrix_B_u has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_B_u.shape[0], self.matrix_B_u.shape[1], n_bmix, n_u))
        if shape_of_callable(self.matrix_B_y) != (n_bmix, n_y):
            raise ValueError('matrix_B_y has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_B_y.shape[0], self.matrix_B_y.shape[1], n_bmix, n_y))
        if shape_of_callable(self.matrix_B_x) != (n_bmix, n_x):
            raise ValueError('matrix_B_x has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_B_u.shape[0], self.matrix_B_x.shape[1], n_bmix, n_x))
        if shape_of_callable(self.vec_B) != (n_bmix, 1):
            raise ValueError('vec_B has wrong dimension')
        #
        if shape_of_callable(self.matrix_start) != (n_bndry, n_y):
            raise ValueError('matrix_start has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_start.shape[0], self.matrix_start.shape[1], n_bndry, n_y))
        if shape_of_callable(self.matrix_end) != (n_bndry, n_y):
            raise ValueError('matrix_end has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_end.shape[0], self.matrix_end.shape[1], n_bndry, n_y))
        if shape_of_callable(self.vec_bndry) != (n_bndry, 1):
            raise ValueError('vec_bndry has wrong dimension')
        if shape_of_callable(self.matrix_u_start) != (n_bndry, n_u):
            raise ValueError('matrix_u_start has wrong dimension({}, {}) should be ({}, {})'.format(
                self.matrix_u_start.shape[0], self.matrix_u_start.shape[1], n_bndry, n_u))
        if shape_of_callable(self.matrix_u_end) != (n_bndry, n_u):
            raise ValueError('marix_u_end has wrong dimension ({}, {}) should be ({}, {})'.format(
                self.matrix_u_end.shape[0], self.matrix_u_end.shape[1], n_bndry, n_u))


    def _set_unset_fields(self):
        """
        Try to complete a poorly described Matrrrices instance
        """
        set_fields = self.__dict__.keys()
        #print('set fields:', set_fields)
        to_be_set_fields = list(set(MAT_FIELDS).difference(set_fields))
        # sort according to MAT_FIELDS
        to_be_set_fields = sorted(to_be_set_fields, key=lambda x: MAT_FIELDS.index(x))  # pylint: disable=unnecessary-lambda
        #print('to be set:', to_be_set_fields)
        qssa_elem = ['smat1', 'smat3', 'f_1']
        dyn_elem = ['smat2', 'smat4', 'f_2']
        bnd_elem = ['lbvec', 'ubvec']
        mix_elem = ['matrix_u', 'matrix_y', 'vec_h', 'mixed_names']
        bmix_elem = ['matrix_B_u', 'matrix_B_y', 'matrix_B_x', 'vec_B']
        bndry_elem = ['matrix_start', 'matrix_end', 'vec_bndry', 'matrix_u_start', 'matrix_u_end']
        for kw_name in to_be_set_fields:
            #print(30*'-')
            #print('Trying to set field: ', kw_name)
            #self.__dict__[kw_name] = _create_default_value(self, kw_name)
            #
            if kw_name in ['y_vec', 'u_vec', 'x_vec']:
                raise ValueError(''.join(['In the construction of a Matrrrices object',
                                          ' at least the fields y_vec, x_vec and u_vec need to',
                                          ' be set, cannot set ', kw_name]))
            n_y = len(self.y_vec)
            n_u = len(self.u_vec)
            n_x = len(self.x_vec)
            if kw_name in ['phi1', 'phi2', 'phi3']:
                self.__dict__[kw_name] = np.zeros((n_y, 1), dtype=float)
            elif kw_name == 'phi1u':
                self.__dict__[kw_name] = np.zeros((n_u, 1), dtype=float)
            # MAYBE: What follows can be abstracted to much simpler code...
            # "QSSA rows" ----------------------------------------------------
            elif kw_name in qssa_elem:
                given_fields = [f for f in set_fields if f in qssa_elem]
                if len(given_fields) == 0:
                    n_qssa = 0
                else:
                    n_qssa = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'smat1':
                    n_cols = n_u
                elif kw_name == 'smat3':
                    n_cols = n_y
                else:
                    n_cols = 1
                self.__dict__[kw_name] = sp.csr_matrix((n_qssa, n_cols))
            # rows of dynamic eqs. -------------------------------------------
            elif kw_name in dyn_elem:
                n_dyn = n_y
                if kw_name == 'smat2':
                    n_cols = n_u
                elif kw_name == 'smat4':
                    n_cols = n_y
                else:
                    n_cols = 1
                self.__dict__[kw_name] = sp.csr_matrix((n_dyn, n_cols))
            # flux bounds ----------------------------------------------------
            elif kw_name in bnd_elem: # This outer loop is just to "improve" readability
                if kw_name == 'lbvec':
                    self.lbvec = -INFINITY*np.ones((n_u, 1))
                else:
                    self.ubvec = INFINITY*np.ones((n_u, 1))
            # rows of mixed ineqs. -------------------------------------------
            elif kw_name in mix_elem:
                given_fields = [f for f in set_fields if f in mix_elem]
                if len(given_fields) == 0:
                    n_mix = 0
                else:
                    n_mix = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'matrix_u':
                    self.matrix_u = sp.csr_matrix((n_mix, n_u))
                elif kw_name == 'matrix_y':
                    self.matrix_y = sp.csr_matrix((n_mix, n_y))
                elif kw_name == 'vec_h':
                    self.vec_h = np.zeros((n_mix, 1))
                else:
                    self.mixed_names = ['mixed_constraint_'+str(i) for i in range(n_mix)]
            # rows of mixed inqs. with Boolean elements ----------------------
            elif kw_name in bmix_elem:
                given_fields = [f for f in set_fields if f in bmix_elem]
                if len(given_fields) == 0:
                    n_bmix = 0
                else:
                    n_bmix = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'matrix_B_u':
                    n_cols = n_u
                elif kw_name == 'matrix_B_y':
                    n_cols = n_y
                elif kw_name == 'matrix_B_x':
                    n_cols = n_x
                else:
                    n_cols = 1
                self.__dict__[kw_name] = sp.csr_matrix((n_bmix, n_cols))
            # rows of boundary conditions ------------------------------------
            elif kw_name in bndry_elem:
                given_fields = [f for f in set_fields if f in bndry_elem]
                if len(given_fields) == 0:
                    n_bndry = 0
                else:
                    n_bndry = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name in ['matrix_start', 'matrix_end']:
                    n_cols = n_y
                elif kw_name == 'vec_bndry':
                    n_cols = 1
                else:
                    n_cols = n_u
                self.__dict__[kw_name] = sp.csr_matrix((n_bndry, n_cols))


    def construct_vectors(self, model):
        """
        construct vectors containing the IDs of dynamical species, reactions,
        and boolean variables, respectively
        """

        # species vector y contains only dynamical species, i.e.:
        y_vec = []
        # dynamical external species
        for ext, macro in model.extracellular_dict.items():
            if not macro['constant'] and not macro['boundaryCondition']:
                y_vec.append(ext)
        # and all macromolecular species
        y_vec = y_vec + list(model.macromolecules_dict.keys())

        # reactions vector u contains all reactions except for degradation reactions
        # Why are these specified by finite kcatforward?
        u_vec = [r for (r, rxn) in model.reactions_dict.items() if np.isfinite(rxn['kcatForward'])]

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

        self.phi1 = np.zeros((len(self.y_vec), 1), dtype=float)

        for m_name, macrom in model.macromolecules_dict.items():
            self.phi1[self.y_vec.index(m_name)] = -macrom['objectiveWeight']

        if phi2:
            self.phi2 = phi2
        else:
            self.phi2 = np.zeros((len(self.y_vec), 1), dtype=float)

        if phi3:
            self.phi3 = phi3
        else:
            self.phi3 = np.zeros((len(self.y_vec), 1), dtype=float)

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
        #FIXME: This is dangerous: If one initial amount is larger one, we never get feasibility
        #       and it is not needed in general
        enforce_biomass = False
        for macrom in model.macromolecules_dict.keys():
            if np.isnan(model.macromolecules_dict[macrom]['initialAmount']):
                enforce_biomass = True
                break
        if enforce_biomass:
            #print('We need to "enforce biomass" here.')
            weights_row = np.zeros(len(self.y_vec), dtype=float)
            for macrom in model.macromolecules_dict.keys():
                weight = float(model.macromolecules_dict[macrom]["molecularWeight"])
                weights_row[self.y_vec.index(macrom)] = weight
            matrix_start = np.append(matrix_start, [weights_row], axis=0)
            vec_bndry = np.append(vec_bndry, [[1.0]], axis=0)

        self.matrix_start = sp.csr_matrix(matrix_start)
        self.matrix_end = sp.csr_matrix(np.zeros((self.matrix_start.shape), dtype=float))
        self.vec_bndry = vec_bndry
        self.matrix_u_start = sp.csr_matrix((vec_bndry.shape[0], len(self.u_vec)), dtype=float)
        self.matrix_u_end = sp.csr_matrix((vec_bndry.shape[0], len(self.u_vec)), dtype=float)


    def construct_reactions(self, model):
        """
        construct matrices S1, S2, S3, S4 and vectors f_1, f_2
        """

        # select rows of species with QSSA
        # # (first entries in stoichiometric matrix belong to extreacellular species,
        # followed by internal metabolites)
        # initialize with indices of internal metabolites
        rows_qssa = list(
            range(len(model.extracellular_dict),
                  len(model.extracellular_dict) + len(model.metabolites_dict)))
        # add non-dynamical extracellualar species
        for row_index, ext in enumerate(model.extracellular_dict.values()):
            if ext['constant'] or ext['boundaryCondition']:
                rows_qssa.append(row_index)

        # select columns of degradation reactions
        cols_deg = []
        for col_index, rxn in enumerate(model.reactions_dict.values()):
            if np.isnan(rxn['kcatForward']): # This is (so far) a convention.
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
        self.f_1 = np.zeros((smat1.shape[0], 1))
        self.f_2 = np.zeros((smat2.shape[0], 1))


    def construct_flux_bounds(self, model):
        """
        construct vectors lb, ub
        """
        lbvec = np.nan*np.ones((len(self.u_vec), 1))
        ubvec = np.nan*np.ones((len(self.u_vec), 1))

        # flux bounds determined by regulation are not considered here
        for index, r_name in enumerate(self.u_vec):
            # lower bounds
            lbvec[index] = float(model.reactions_dict[r_name].get('lowerFluxBound', -INFINITY))
            if not model.reactions_dict[r_name]['reversible']:
                lbvec[index] = 0.0
            ubvec[index] = float(model.reactions_dict[r_name].get('upperFluxBound', INFINITY))

        self.lbvec = lbvec
        self.ubvec = ubvec

    def construct_fullmixed(self, model):
        """
        construct matrices for regulation
        """

        epsilon = EPSILON
        big_u = BIGM
        big_l = MINUSBIGM

        # control of discrete jumps

        # initialize matrices
        n_assignments = sum([len(evnt['listOfAssignments'])
                             for evnt in model.events_dict.values()])
        y_matrix_1 = np.zeros((n_assignments, len(self.y_vec)), dtype=float)
        u_matrix_1 = np.zeros((n_assignments, len(self.u_vec)), dtype=float)
        x_matrix_1 = np.zeros((n_assignments, len(self.x_vec)), dtype=float)
        b_vec_1 = np.zeros((n_assignments, 1))

        for event_index, event in enumerate(model.events_dict.values()):
            variables = event['variable'].split(' + ')
            # Difference between geq and gt??
            if event['relation'] == 'geq' or event['relation'] == 'gt':
                for i, affected_bool in enumerate(event['listOfAssignments']):
                    for variable in variables:
                        # boolean variable depends on species amount
                        if variable in self.y_vec:
                            species_index = self.y_vec.index(variable)
                            y_matrix_1[event_index, species_index] = 1
                            if event['listOfEffects'][i] == 0:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = \
                                                                                    epsilon + big_u
                                b_vec_1[event_index] = event['threshold'] + big_u
                            elif event['listOfEffects'][i] == 1:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = \
                                                                                - (epsilon + big_u)
                                b_vec_1[event_index] = event['threshold'] - epsilon

                        # boolean variable depends on flux
                        elif variable in self.u_vec:
                            flux_index = self.u_vec.index(variable)
                            u_matrix_1[event_index, flux_index] = 1
                            if event['listOfEffects'][i] == 0:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = \
                                                                                    epsilon + big_u
                                b_vec_1[event_index] = event['threshold'] + big_u
                            elif event['listOfEffects'][i] == 1:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = \
                                                                                - (epsilon + big_u)
                                b_vec_1[event_index] = event['threshold'] - epsilon
                        else:
                            print(variable + ' not defined as Species or Reaction!')


            # TODO Difference between leq and lt??
            elif event['relation'] == 'leq' or event['relation'] == 'lt':
                for i, affected_bool in enumerate(event['listOfAssignments']):
                    for variable in variables:
                        # boolean variable depends on species amount
                        if variable in self.y_vec:
                            species_index = self.y_vec.index(variable)
                            y_matrix_1[event_index, species_index] = -1
                            if event['listOfEffects'][i] == 0:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = -big_l
                                b_vec_1[event_index] = -event['threshold'] - big_l
                            elif event['listOfEffects'][i] == 1:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = big_l
                                b_vec_1[event_index] = -event['threshold']

                        # boolean variable depends on flux
                        elif variable in self.u_vec:
                            flux_index = self.u_vec.index(variable)
                            u_matrix_1[event_index, flux_index] = -1
                            if event['listOfEffects'][i] == 0:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = -big_l
                                b_vec_1[event_index] = -event['threshold'] - big_l
                            elif event['listOfEffects'][i] == 1:
                                x_matrix_1[event_index, self.x_vec.index(affected_bool)] = big_l
                                b_vec_1[event_index] = -event['threshold']
                        else:
                            print(variable + ' not defined as Species or Reaction!')
                event_index += 1


        n_rules = 0
        for rule in model.rules_dict.keys():
            if 'reactionID' in model.rules_dict[rule]:
                n_rules += 1
            elif 'operator' in model.rules_dict[rule]:
                n_rules += len(model.rules_dict[rule]['indicators']) + 1

        y_matrix_2 = np.zeros((n_rules, len(self.y_vec)), dtype=float)
        u_matrix_2 = np.zeros((n_rules, len(self.u_vec)), dtype=float)
        x_matrix_2 = np.zeros((n_rules, len(self.x_vec)), dtype=float)
        b_vec_2 = np.zeros((n_rules, 1))

        rule_row_index = 0

        for rule_name, rule in model.rules_dict.items():
            # Control of continuous dynamics by discrete states
            if 'reactionID' in rule:
                rxn_index = self.u_vec.index(rule['reactionID'])
                par_index = self.x_vec.index(rule['bool_parameter'])
                if rule['direction'] == 'lower':
                    u_matrix_2[rule_row_index, rxn_index] = -1
                    if np.isnan(rule['threshold']):
                        x_matrix_2[rule_row_index, par_index] = big_l
                    else:
                        x_matrix_2[rule_row_index, par_index] = float(rule['threshold'])
                if rule['direction'] == 'upper':
                    u_matrix_2[rule_row_index, rxn_index] = 1
                    if np.isnan(rule['threshold']):
                        x_matrix_2[rule_row_index, par_index] = -big_u
                    else:
                        x_matrix_2[rule_row_index, par_index] = float(rule['threshold'])
                rule_row_index += 1 # only requires one line, i.e. one inequality
            # Boolean Algebra Rules
            elif 'operator' in rule:
                variable_index = self.x_vec.index(rule_name)
                # Discriminate AND and OR
                if rule['operator'] == 304: # AND
                    # first inequality (the one containing all variables)
                    x_matrix_2[rule_row_index, variable_index] = -1
                    b_vec_2[rule_row_index] = len(rule['indicators']) - 1
                    for i in range(len(rule['indicators'])):
                        indicator_index = self.x_vec.index(rule['indicators'][i])
                        # positive in the first inequality containing all variables
                        x_matrix_2[rule_row_index, indicator_index] = 1
                        # negative in another (dependent variable is positive there)
                        x_matrix_2[rule_row_index + i + 1, indicator_index] = -1
                        x_matrix_2[rule_row_index + i + 1, variable_index] = 1
                    rule_row_index += len(rule['indicators']) + 1
                elif rule['operator'] == 306: # OR
                    # first inequality (the one containing all variables)
                    x_matrix_2[rule_row_index, variable_index] = 1
                    for i in range(len(rule['indicators'])):
                        indicator_index = self.x_vec.index(rule['indicators'][i])
                        # negative in the first inequality containing all variables
                        x_matrix_2[rule_row_index, indicator_index] = -1
                        # positive in another (dependent variable is positive there)
                        x_matrix_2[rule_row_index + i + 1, indicator_index] = 1
                        x_matrix_2[rule_row_index + i + 1, variable_index] = -1
                    rule_row_index += len(rule['indicators']) + 1


        self.matrix_B_y = sp.csr_matrix(np.vstack((y_matrix_1, y_matrix_2)))
        self.matrix_B_u = sp.csr_matrix(np.vstack((u_matrix_1, u_matrix_2)))
        self.matrix_B_x = sp.csr_matrix(np.vstack((x_matrix_1, x_matrix_2)))
        self.vec_B = np.vstack((b_vec_1, b_vec_2))

    def construct_mixed(self, model):
        """
        constructs matrices H_y and H_u
        """
        # TODO: What if more than one enzyme catalyzes one reactions? One should at least check.
        # check whether the model contains quota compounds
        # QUESTION: And where do we check/need the 'storage' property?
        matrix_y_1, quota_constraint_names = self.construct_quota(model)
        matrix_u_1 = np.zeros((matrix_y_1.shape[0], len(self.u_vec)), dtype=float)

        # enzyme capacity constraints
        matrix_y_2, matrix_u_2, enzcap_names = self.construct_enzcap(model)

        # maintenance constraints
        matrix_y_3, matrix_u_3, maint_constraint_names = self.construct_maintenance(model)

        # stacking of the resulting matrices
        self.matrix_u = sp.csr_matrix(np.vstack((matrix_u_1, matrix_u_2, matrix_u_3)))
        self.matrix_y = sp.csr_matrix(np.vstack((matrix_y_1, matrix_y_2, matrix_y_3)))
        # h always contains only zeros
        self.vec_h = np.zeros((self.matrix_u.shape[0], 1), dtype=float)
        self.mixed_names = quota_constraint_names + enzcap_names + maint_constraint_names


    def construct_quota(self, model):
        """
        Construct the H_B matrix ('y_matrix' matrix of quota constraints)
        """
        all_q_names = [n for (n, m) in model.macromolecules_dict.items() \
                                                                    if m['speciesType'] == 'quota']
        #According to https://doi.org/10.1016/j.jtbi.2020.110317 this is a single constraint.
        if all_q_names:
            all_q_names = [all_q_names[0]]
        n_quota = len(all_q_names)
        y_matrix = np.zeros((n_quota, len(self.y_vec)))
        quota_constraint_names = n_quota*['quota']
        if n_quota:
            biom_perc = model.macromolecules_dict[all_q_names[0]]['biomassPercentage']
            # TODO: (a) It's not 'percentage', (b) what if =1?, (c) what if multiple quota present?
            for m_name, macro in model.macromolecules_dict.items():
                y_matrix[0, self.y_vec.index(m_name)] = macro['molecularWeight']*biom_perc
            for q_name in all_q_names:
                # (b-1)*w = ( (b-1)/b ) * (b*w)
                y_matrix[0, self.y_vec.index(q_name)] *= (biom_perc-1.0)/biom_perc
        return y_matrix, quota_constraint_names


    def construct_enzcap(self, model):
        """
        Construct matrices for enzyme capacity constraints: H_C ~> u_matrix and
                                             filter matrix H_E ~> y_matrix
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
        u_matrix = np.zeros((n_rows, len(self.u_vec)), dtype=float)
        y_matrix = np.zeros((n_rows, len(self.y_vec)), dtype=float)
        enzcap_names = n_rows*['enzyme_cap_']

        # fill matrices
        e_cnt = 0  # enzyme counter
        i = 0  # row counter

        # iterate over enzymes
        for enzyme in model.macromolecules_dict.keys():
            # if macromolecule doesn't catalyze any reaction (e.g. transcription factors), it won't be regarded
            enzyme_catalyzes_anything = False
            # n_rev contains a number for each catalytically active enzyme
            if e_cnt < len(n_rev):
                # increment macromolecule counter
                c_rev = 0  # reversible-reaction-per-enzyme counter
                # iterate over reactions
                if c_rev <= n_rev[e_cnt]:
                    for r_name, rxn in model.reactions_dict.items():
                        # if there is a reaction catalyzed by this macromolecule (i.e. it is a true enzyme)
                        if rxn['geneProduct'] == enzyme:
                            enzyme_catalyzes_anything = True
                            # reversible reactions
                            if rxn['reversible']:
                                # boolean variable specifies whether to include forward or backward k_cat
                                fwd = True
                                # in order to cover all possible combinations of reaction fluxes
                                for r_cnt in range(2 ** n_rev[e_cnt]):
                                    if fwd:
                                        u_matrix[i + r_cnt, self.u_vec.index(r_name)] = \
                                                                  np.reciprocal(rxn['kcatForward'])
                                        y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = 1
                                        r_cnt += 1
                                        # true after half of the combinations for the first reversible reaction
                                        # true after 1/4 of the combinations for the second reversible reaction
                                        # and so on.
                                        if np.mod(r_cnt, 2 ** n_rev[e_cnt] / 2 ** (c_rev + 1)) == 0:
                                            fwd = False
                                    else:
                                        u_matrix[i + r_cnt, self.u_vec.index(r_name)] = \
                                                              -1*np.reciprocal(rxn['kcatBackward'])
                                        y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = -1
                                        r_cnt += 1
                                        # as above, fwd will be switched after 1/2, 1/4, ... of the possible combinations
                                        if np.mod(r_cnt, 2**n_rev[e_cnt] / 2**(c_rev + 1)) == 0:
                                            fwd = True
                                    enzcap_names[i + r_cnt] += enzyme + '_' + str(r_cnt)
                                c_rev += 1
                            # irreversible reactions
                            else:
                                # simply enter 1/k_cat for each combination
                                # (2^0 = 1 in case of an enzyme that only catalyzes irreversible reactions)
                                for r_cnt in range(2 ** n_rev[e_cnt]):
                                    u_matrix[i + r_cnt, self.u_vec.index(r_name)] = np.reciprocal(
                                        rxn['kcatForward'])
                                    y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = 1
                                    enzcap_names[i + r_cnt] += enzyme + '_' + str(r_cnt)
            if enzyme_catalyzes_anything:
                i += 2 ** n_rev[e_cnt]
                e_cnt += 1
                #print(e_cnt, enzyme)# DEBUG!!!
        #print(-HE_matrix[:4,:4])# DEBUG
        #print(HC_matrix[:4,:4])# DEBUG
        return -y_matrix, u_matrix, enzcap_names

    def construct_maintenance(self, model):
        """
        Constructs the maintenance matrices
        """
        # TODO: Why did we (before) assume that there is only one maint.-reaction?
        # How many maintenance reactions are there?
        maint_rxns = [n for (n, r) in model.reactions_dict.items() if r['maintenanceScaling'] > 0]
        #print(maint_rxns)#
        maint_constraint_names = ['maintenance_'+r for r in maint_rxns]
        u_matrix = np.zeros((len(maint_rxns), len(self.u_vec)))
        y_matrix = np.zeros((len(maint_rxns), len(self.y_vec)))
        for k, main_rxn in enumerate(maint_rxns):
            main_scaling = model.reactions_dict[main_rxn]['maintenanceScaling']
            main_index = self.u_vec.index(main_rxn)
            u_matrix[k, main_index] = -1.0
            for i, macrom in enumerate(model.macromolecules_dict.values()):
                y_matrix[k, i] = main_scaling * macrom['molecularWeight']
        #
        return y_matrix, u_matrix, maint_constraint_names
