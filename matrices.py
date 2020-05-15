import numpy as np

def construct_HcHe(model):
    """
    Construct matrices for enzyme capacity constraints: H_C and filter matrix H_E
    """
    # calculate number of rows in H_C and H_E:
    n_rev = []  # list containing number of reversible rxns per enzyme

    # iterate over enzymes
    for enzyme in model.species_dict.keys():
        e_rev = 0  # number of reversible reactions catalyzed by this enzyme
        if model.species_dict[enzyme]['speciesType'] == 'enzyme':
            enzyme_catalyzed_anything = False
            # iterate over reactions
            for rxn, key in model.reactions_dict.items():
                if model.reactions_dict[rxn]['geneProduct'] == enzyme:
                    enzyme_catalyzed_anything = True
                    if model.reactions_dict[rxn]['reversible']:
                        e_rev = e_rev + 1
            if enzyme_catalyzed_anything:
                n_rev.append(e_rev)

    n = sum(2 ** i for i in n_rev)  # number of rows in H_C and H_E

    # initialize matrices
    model.HC_matrix = np.zeros((n, len(model.reactions_dict)))
    model.HE_matrix = np.zeros((n, len(n_rev)))  # number of enzymes = number of columns in H_E

    # fill matrices
    e = 0  # enzyme counter
    i = 0  # row counter

    # iterate over enzymes
    for enzyme in model.species_dict.keys():
        enzyme_catalyzes_anything = False
        if e < len(n_rev):
            if model.species_dict[enzyme]['speciesType'] == 'enzyme':
                j = 0  # reversible-reaction-per-enzyme counter
                # iterate over reactions
                if j <= n_rev[e]:
                    for rxn, key in model.reactions_dict.items():
                        if model.reactions_dict[rxn]['geneProduct'] == enzyme:
                            enzyme_catalyzes_anything = True
                            if model.reactions_dict[rxn]['reversible']:
                                fwd = True
                                for r in range(2 ** n_rev[e]):
                                    if fwd:
                                        model.HC_matrix[i + r][
                                            list(model.reactions_dict.keys()).index(rxn)] = np.reciprocal(
                                            model.reactions_dict[rxn]['kcatForward'])
                                        r = r + 1
                                        if np.mod(r, 2 ** n_rev[e] / 2 ** (j + 1)) == 0:
                                            fwd = False
                                    else:
                                        model.HC_matrix[i + r][
                                            list(model.reactions_dict.keys()).index(rxn)] = -1 * np.reciprocal(
                                            model.reactions_dict[rxn]['kcatBackward'])
                                        r = r + 1
                                        if np.mod(r, 2 ** n_rev[e] / 2 ** (j + 1)) == 0:
                                            fwd = True
                                j = j + 1
                            else:  # irreversible
                                for r in range(2 ** n_rev[e]):
                                    model.HC_matrix[i + r][
                                        list(model.reactions_dict.keys()).index(rxn)] = np.reciprocal(
                                        model.reactions_dict[rxn]['kcatForward'])
        if enzyme_catalyzes_anything:
            i = i + 2 ** n_rev[e]
            e = e + 1



    def __construct_Hm(self):
        """
        Constructs the HM matrix
        """
        if self.maintenance_percentage:
            self.HM_matrix = np.zeros((len(self.maintenance_percentage), len(self.reactions)))
            for row, reac in enumerate(self.maintenance_percentage.keys()):
                self.HM_matrix[row, self.reactions.index(reac)] = 1.0 / self.maintenance_percentage[reac]
                if reac in self.protein_reactions:
                    self.parameter_list[self.param_main[reac]] = [['HM[' + str(row) + ',' + str(self.reactions.index(reac)) + ']'],
                                                                  self.maintenance_percentage[reac], [-1], [1], 'maintenance']
                else:
                    self.parameter_list[self.param_main[reac]] = [
                        ['HM[' + str(row) + ',' + str(self.reactions.index(reac)) + ']'],
                        self.maintenance_percentage[reac], [-1], [self.scale], 'maintenance']


    def __construct_Hb(self):
        """
        Construct the HB matrix
        """
        for specie_w in self.param_ob_weight:
            if not specie_w in self.param_weight:
                self.parameter_list[self.param_ob_weight[specie_w]] = [
                    ['objective[' + str(self.species.index(specie_w)) + ']'], self.objective_weight[specie_w], [1], [1],
                    'objective']
            else:
                self.parameter_list[self.param_ob_weight[specie_w]] = [
                    ['objective[' + str(self.species.index(specie_w)) + ']'], self.objective_weight[specie_w], [1], [1],
                    'ob_weight']
        for specie_w in self.param_weight:
            if not specie_w in self.param_ob_weight:
                self.parameter_list[self.param_weight[specie_w]] = [[], self.molecular_weight[specie_w], [], [],
                                                                    'weight']
        if self.biomass_percentage:
            self.HB_matrix = np.zeros((len(self.biomass_percentage), len(self.species)))
            biomass_vector = np.zeros(len(self.species))
            for j, specie in enumerate(self.species[-self.numbers['proteins']:]):
                biomass_vector[j - self.numbers['proteins']] = self.molecular_weight[specie]
            for i in self.param_biomp.keys():
                self.parameter_list[self.param_biomp[i]] = [[], self.biomass_percentage[i], [], [], 'biom_percent']
            for row, specie in enumerate(self.biomass_percentage.keys()):
                for specie_w in self.param_weight:
                    self.parameter_list[self.param_weight[specie_w]][0].append(
                        'HB[' + str(row) + ',' + str(self.species.index(specie_w)) + ']')
                    self.parameter_list[self.param_weight[specie_w]][2].append(1)
                    if specie_w == specie:
                        self.parameter_list[self.param_weight[specie_w]][3].append(self.biomass_percentage[specie] - 1)
                    else:
                        self.parameter_list[self.param_weight[specie_w]][3].append(self.biomass_percentage[specie])
                for coloumn in range(self.numbers['proteins']):
                    self.HB_matrix[row, coloumn - self.numbers['proteins']] = self.molecular_weight[
                                                                                  self.species[coloumn - self.numbers['proteins']]] * \
                                                                              self.biomass_percentage[specie]
                    try:
                        self.parameter_list[self.param_biomp[specie]][0].append(
                            'HB[' + str(row) + ',' + str(coloumn - self.numbers['proteins']) + ']')
                        self.parameter_list[self.param_biomp[specie]][2].append(1)
                        if specie == self.species[coloumn - self.numbers['proteins']]:
                            self.HB_matrix[row, coloumn - self.numbers['proteins']] = (self.biomass_percentage[specie] - 1) * \
                                                                                      self.molecular_weight[specie]
                            self.parameter_list[self.param_biomp[specie]][3].append(-1)
                        else:
                            self.parameter_list[self.param_biomp[specie]][3].append(1)
                    except KeyError:
                        if specie == self.species[coloumn - self.numbers['proteins']]:
                            self.HB_matrix[row, coloumn - self.numbers['proteins']] = (self.biomass_percentage[specie] - 1) * \
                                                                                  self.molecular_weight[specie]