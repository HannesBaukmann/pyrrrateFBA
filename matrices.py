# because a single enzyme can catalyze multiple reactions, we dont get a 1-to-1 mapping and need the inverse dict as well
self.enzyme_catalyzes = {}
for j, reactionname in enumerate(reaction_catalyzed_by):
    try:
        self.enzyme_catalyzes[reaction_catalyzed_by[reactionname]].append(reactionname)
    except KeyError:
        self.enzyme_catalyzes[reaction_catalyzed_by[reactionname]] = [reactionname]
# Construct HC, HE matrices
# eliminate the empty set from the dict
self.enzyme_catalyzes = {k: v for k, v in list(self.enzyme_catalyzes.items()) if v != []}
# Go through enzymes and construct the enzyme capacity constraints accordingly
for enz, reac_array in list(self.enzyme_catalyzes.items()):
    reversible_reactions = []
    irreversible_reactions = []
    for reac in reac_array:
        if reac in self.reversible_fluxes:
            reversible_reactions.append(reac)
        else:
            irreversible_reactions.append(reac)
    self.__constructHCHEmatrix(enz, reversible_reactions, irreversible_reactions)

self.__constructHBmatrix()
self.__constructHMmattrix()


def __constructHCHEmatrix(self, enz, reversible_reactions, irreversible_reactions):
    """
    Construct the enzyme capacity constraints iteratively.
    Adds the constraints for the enzyme enz to the already present HC, HE matrices.
    """
    # include forward kcat values for the irreversible reactions
    new_row_number = 2 ** len(reversible_reactions)
    new_HC = np.zeros((new_row_number, len(self.reactions)))
    new_HCtemp = np.zeros((1, len(self.reactions)))
    # check if HC is already present.
    if not hasattr(self.HC_matrix, 'shape'):
        HC_size = 0
    else:
        HC_size = self.HC_matrix.shape[0]
    # First we set the elements for the irreversible reactions.
    # While RAM enforces the backward kcat for irreversible reactions to be zero. This method can handle irreversible reactions with
    # k_forward = 0 and k_backward != 0.
    for irr in irreversible_reactions:
        if not self.kcat_values[irr][0] == 0:
            # Decide whether the parameter will be scaled in the deFBA model.
            if irr in self.protein_reactions:
                self.parameter_list[self.param_kcat[irr][0]] = [
                    ['HC[' + str(HC_size) + ',' + str(self.reactions.index(irr)) + ']'], self.kcat_values[irr][0],
                    [-1], [1], 'kcat']
            else:
                self.parameter_list[self.param_kcat[irr][0]] = [
                    ['HC[' + str(HC_size) + ',' + str(self.reactions.index(irr)) + ']'], self.kcat_values[irr][0],
                    [-1], [self.scale], 'kcat']
            # Add all positions of the parameter
            for i in range(1, new_row_number):
                self.parameter_list[self.param_kcat[irr][0]][0].append(
                    'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(irr)) + ']')
                self.parameter_list[self.param_kcat[irr][0]][2].append(-1)
                if irr in self.protein_reactions:
                    self.parameter_list[self.param_kcat[irr][0]][3].append(1)
                else:
                    self.parameter_list[self.param_kcat[irr][0]][3].append(self.scale)
            # Put the values in
            new_HCtemp[0, self.reactions.index(irr)] = 1.0 / (self.kcat_values[irr][0])
        # handle backward pointing reactions
        elif not self.kcat_values[irr][1] == 0:
            if irr in self.protein_reactions:
                self.parameter_list[self.param_kcat[irr][1]] = [
                    ['HC[' + str(HC_size) + ',' + str(self.reactions.index(irr)) + ']'], self.kcat_values[irr][1],
                    [-1], [-1], 'kcat']
            else:
                self.parameter_list[self.param_kcat[irr][1]] = [
                    ['HC[' + str(HC_size) + ',' + str(self.reactions.index(irr)) + ']'], self.kcat_values[irr][1],
                    [-1], [-self.scale], 'kcat']
            for i in range(1, new_row_number):
                self.parameter_list[self.param_kcat[irr][1]][0].append(
                    'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(irr)) + ']')
                self.parameter_list[self.param_kcat[irr][1]][2].append(-1)
                if irr in self.protein_reactions:
                    self.parameter_list[self.param_kcat[irr][1]][3].append(-1)
                else:
                    self.parameter_list[self.param_kcat[irr][1]][3].append(-self.scale)
            new_HCtemp[0, self.reactions.index(irr)] = -1.0 / (self.kcat_values[irr][1])
    # state which enzyme is in play
    new_HE = np.zeros((new_row_number, len(self.species)))
    new_HEtemp = np.zeros((1, len(self.species)))
    new_HEtemp[0, self.species.index(enz)] = 1
    # generate templates
    for rows in range(2 ** (len(reversible_reactions))):
        new_HC[rows, :] = new_HCtemp
        new_HE[rows, :] = new_HEtemp
    # first column
    if reversible_reactions:
        for rev_reac in reversible_reactions:
            self.parameter_list[self.param_kcat[rev_reac][0]] = [[], self.kcat_values[rev_reac][0], [], [], 'kcat']
            self.parameter_list[self.param_kcat[rev_reac][1]] = [[], self.kcat_values[rev_reac][1], [], [], 'kcat']
        for i in range(new_row_number):
            if (-1) ** i == 1:
                self.parameter_list[self.param_kcat[reversible_reactions[0]][0]][0].append(
                    'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(reversible_reactions[0])) + ']')
                self.parameter_list[self.param_kcat[reversible_reactions[0]][0]][2].append(-1)
                if reversible_reactions[0] in self.protein_reactions:
                    self.parameter_list[self.param_kcat[reversible_reactions[0]][0]][3].append(1)
                else:
                    self.parameter_list[self.param_kcat[reversible_reactions[0]][0]][3].append(self.scale)
                new_HC[i, self.reactions.index(reversible_reactions[0])] = 1.0 / (
                    self.kcat_values[reversible_reactions[0]][0])
            else:
                self.parameter_list[self.param_kcat[reversible_reactions[0]][1]][0].append(
                    'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(reversible_reactions[0])) + ']')
                self.parameter_list[self.param_kcat[reversible_reactions[0]][1]][2].append(-1)
                if reversible_reactions[0] in self.protein_reactions:
                    self.parameter_list[self.param_kcat[reversible_reactions[0]][1]][3].append(-1)
                else:
                    self.parameter_list[self.param_kcat[reversible_reactions[0]][1]][3].append(-self.scale)
                new_HC[i, self.reactions.index(reversible_reactions[0])] = -1.0 / (
                    self.kcat_values[reversible_reactions[0]][1])
        for j in range(1, len(reversible_reactions)):
            for i in range(new_row_number):
                if not ((i % (2 ** (j + 1))) < 2 ** j):
                    self.parameter_list[self.param_kcat[reversible_reactions[j]][1]][0].append(
                        'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(reversible_reactions[j])) + ']')
                    self.parameter_list[self.param_kcat[reversible_reactions[j]][1]][2].append(-1)
                    if reversible_reactions[j] in self.protein_reactions:
                        self.parameter_list[self.param_kcat[reversible_reactions[j]][1]][3].append(-1)
                    else:
                        self.parameter_list[self.param_kcat[reversible_reactions[j]][1]][3].append(-self.scale)
                    new_HC[i, self.reactions.index(reversible_reactions[j])] = -1.0 / (
                        self.kcat_values[reversible_reactions[j]][1])
                else:
                    self.parameter_list[self.param_kcat[reversible_reactions[j]][0]][0].append(
                        'HC[' + str(HC_size + i) + ',' + str(self.reactions.index(reversible_reactions[j])) + ']')
                    self.parameter_list[self.param_kcat[reversible_reactions[j]][0]][2].append(-1)
                    if reversible_reactions[j] in self.protein_reactions:
                        self.parameter_list[self.param_kcat[reversible_reactions[j]][0]][3].append(1)
                    else:
                        self.parameter_list[self.param_kcat[reversible_reactions[j]][0]][3].append(self.scale)
                    new_HC[i, self.reactions.index(reversible_reactions[j])] = 1.0 / (
                        self.kcat_values[reversible_reactions[j]][0])
    # if no the method is called the first time, create a HC matrix
    if not hasattr(self.HC_matrix, 'shape'):
        self.HC_matrix = new_HC
        self.HE_matrix = new_HE
    # else add the new constraints
    else:
        self.HC_matrix = np.r_[self.HC_matrix, new_HC]
        self.HE_matrix = np.r_[self.HE_matrix, new_HE]



    def __constructHMmattrix(self):
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


    def __constructHBmatrix(self):
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

