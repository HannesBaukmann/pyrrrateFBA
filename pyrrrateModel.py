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

        # Create an empty variable to save the latest results of the simulation methods
        self.results = None

        # Set the name of the model
        self.name = name

        def print_numbers():
            pass
            # species
            # - metabolites
            # - intra
            # - extra
            # - macromolecules
            # - enzymes
            # - RE
            # - NRE
            # - TF
            # - quota
            # - storage
            # reactions
            # - uptake
            # - met
            # - translation
            # - spontaneous
            # - maintenance
            # regulation
            # - rules
            # - Regulatory P
            # - regulated reactions

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