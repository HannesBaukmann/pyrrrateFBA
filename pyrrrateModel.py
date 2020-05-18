class Model(object):
    """
    This class represents a biochemical reaction network, specifically the ODE of the form
    zdot = S * v(z), where S is the stoichiometric matrix, v the reaction rate vector, and
    x the species concentration vector.
    The reactions vector consists of v=(v_y, v_x, v_p), with exchange reactions v_y,
    metabolite transformations v_x, and biomass/enzyme production v_p.
    Species vector consists of z=(y,x,p), with external species y, metabolites x, and
    proteins p.
    The information about catalyzing enzymes is encoded via H_c * v <= H_E * p.
    Biomass composition constraint H_B * x <=0;
    Maintenance constraints H_A *v \geq H_F p
    """

    def __init__(self, stoich, name, metabolites, macromolecules, reactions, HC=None, HE=None, HB=None, HM=None):
        """
        DeFbaModel constructor.

        Required arguments:


        Optional arguments:
        - HC                            enzyme capacity constraint (ECC) matrix
        - HE                            filter matrix to determine ECC constraint
        - HB                            biomass composition matrix
        - HM                            maintenance constraint matrix
        """

        self.stoich = stoich
        self.name = name
        self.metabolites_dict = metabolites
        self.macromolecules_dict = macromolecules
        self.reactions_dict = reactions
        self.HC_matrix=HC
        self.HE_matrix=HE
        self.HB_matrix=HB
        self.HM_matrix=HM

        # Create an empty variable to save the latest results of the simulation methods
        self.results = None

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


        def FBA(self, objective=None):
            if not objective:
                # default objectiv is biomass function
                # search for biomass reaction
                # if not found -> Error
                pass
            else:
                # check whether rxn exists
                pass
            # if vmin, vmas aren't given for a reaction, set it to -1000,1000