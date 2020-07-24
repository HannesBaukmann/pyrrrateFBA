import numpy as np

def construct_HcHe(model):
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

    n = sum(2 ** i for i in n_rev)  # number of rows in H_C and H_E

    # initialize matrices
    # number of columns in HC matrix = number of reactions, minus degradation reactions
    m_HC = len(model.reactions_dict)
    for rxn in model.reactions_dict.keys():
        if np.isnan(model.reactions_dict[rxn]['kcatForward']):
            m_HC = m_HC - 1

    # number of columns in HE matrix = number of macromolecules, plus dynamic extracellular species
    # n_dynamic_extracellular is required as an offset (extracellular metabolites cannot catalyze reactions)
    n_dynamic_extracellular = len(model.extracellular_dict)
    for ext in model.extracellular_dict.keys():
        if model.extracellular_dict[ext]['constant'] or model.extracellular_dict[ext]['boundaryCondition']:
            n_dynamic_extracellular = n_dynamic_extracellular - 1
    m_HE = len(model.macromolecules_dict) + n_dynamic_extracellular

    model.HC_matrix = np.zeros((n, m_HC))
    model.HE_matrix = np.zeros((n, m_HE))

    # fill matrices
    j = 0  # macromolecule counter
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
                for rxn, key in model.reactions_dict.items():
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
                                    model.HC_matrix[i + r][
                                        list(model.reactions_dict.keys()).index(rxn)] = np.reciprocal(
                                        model.reactions_dict[rxn]['kcatForward'])
                                    model.HE_matrix[i + r][j+n_dynamic_extracellular] = 1
                                    r = r + 1
                                    # true after half of the combinations for the first reversible reaction
                                    # true after 1/4 of the combinations for the second reversible reaction
                                    # and so on.
                                    if np.mod(r, 2 ** n_rev[e] / 2 ** (c_rev + 1)) == 0:
                                        fwd = False
                                else:
                                    model.HC_matrix[i + r][
                                        list(model.reactions_dict.keys()).index(rxn)] = -1 * np.reciprocal(
                                        model.reactions_dict[rxn]['kcatBackward'])
                                    model.HE_matrix[i + r][j+n_dynamic_extracellular] = -1
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
                                model.HC_matrix[i + r][
                                    list(model.reactions_dict.keys()).index(rxn)] = np.reciprocal(
                                    model.reactions_dict[rxn]['kcatForward'])
                                model.HE_matrix[i + r][j+n_dynamic_extracellular] = -1
        if enzyme_catalyzes_anything:
            i = i + 2 ** n_rev[e]
            e = e + 1
        j = j + 1 # next macromolecule


def construct_Hm(model):
    """
    Constructs the H_M matrix
    """
    model.HM_matrix = np.zeros((len(model.reactions_dict), len(model.macromolecules_dict)))

    for rxn in model.reactions_dict.keys():
        if model.reactions_dict[rxn]['maintenanceScaling'] > 0:
            for mm in model.macromolecules_dict.keys():
                model.HM_matrix[list(model.reactions_dict.keys()).index(rxn)][
                    list(model.macromolecules_dict.keys()).index(mm)] = model.reactions_dict[rxn]['maintenanceScaling'] * model.macromolecules_dict[mm]['initialAmount']


def construct_Hb(model):
    """
    Construct the H_B matrix
    """
    # count number of quota and storage compounds
    b = 0
    for mm in model.macromolecules_dict.keys():
        if model.macromolecules_dict[mm]['speciesType'] == 'quota':
            b += 1

    if b > 0:
        model.HB_matrix = np.zeros((b, len(model.macromolecules_dict)))
        r = 0  # row (quota) counter
        if r < b:
            for q in model.macromolecules_dict.keys():
                if model.macromolecules_dict[q]['speciesType'] == 'quota':
                    for mm in model.macromolecules_dict.keys():
                        if mm == q:
                            model.HB_matrix[r][list(model.macromolecules_dict.keys()).index(mm)] = \
                            (model.macromolecules_dict[q]['biomassPercentage'] - 1) * model.macromolecules_dict[mm]['molecularWeight']
                        else:
                            model.HB_matrix[r][list(model.macromolecules_dict.keys()).index(mm)] = \
                            model.macromolecules_dict[q]['biomassPercentage'] * model.macromolecules_dict[mm]['molecularWeight']
                    r += 1
    else:
        print('The model does not seem to include quota compouds with the mandatory biomass percentage defined. Therefore, no HB matrix will be constructed.')
        # take note that model.HB_matrix may not exist for some models!